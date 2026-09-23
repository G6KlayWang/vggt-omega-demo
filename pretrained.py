"""Optional real RGB reconstruction: credited Omega backbone + our unprojection.

The from-scratch assignment contribution lives in core.py, not this adapter.
No checkpoint download; uses the user's existing local weight file.
"""
import argparse
import base64
import io
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from geometry import unproject, write_ply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("vggt_omega_1b_512.pt"))
    p.add_argument("--images", nargs="+", type=Path)
    p.add_argument("--video", type=Path, help="Optional video instead of the bundled sample images")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/real"))
    p.add_argument("--filter-white-bg", action="store_true", help="Remove points with all RGB channels >240/255 from PLY and viewer")
    p.add_argument("--filter-black-bg", action="store_true", help="Remove points with RGB sum <16/255 from PLY and viewer")
    args = p.parse_args()
    if args.frames < 2 or args.resolution < 64 or args.resolution % 16:
        p.error("Use at least two frames and a resolution >=64 divisible by 16")
    if not args.checkpoint.is_file():
        p.error(f"Checkpoint not found: {args.checkpoint}")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(__file__).parent / "upstream"))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    args.out.mkdir(parents=True, exist_ok=True)
    paths = args.images
    if paths is None and args.video is None:
        paths = sorted((Path(__file__).parent / "data" / "sample").glob("*.jpg"))
    if paths is None:
        import cv2
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            p.error(f"Could not open video: {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, min(total - 1, max(1, int(fps))), args.frames).astype(int)
        paths = []
        for i, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode video frame {index}")
            path = args.out / f"input_{i:02d}.png"
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
            paths.append(path)
        capture.release()
    if len(paths) < 2:
        p.error("Provide at least two overlapping views")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info("Loading local weights with mmap; device=%s, resolution=%d", device, args.resolution)
    start = time.perf_counter()
    # Avoid allocating a second 4.6 GB parameter copy during initialization.
    with torch.device("meta"):
        model = VGGTOmega(autocast=device == "cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    # Materialize any nonpersistent normalization buffers without a second model.
    meta_buffers = [(n, b) for n, b in model.named_buffers() if b.is_meta]
    if meta_buffers:
        logging.info("Materializing %d nonpersistent buffers", len(meta_buffers))
        for name, buffer in meta_buffers:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            if attr == "_resnet_mean":
                value = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            elif attr == "_resnet_std":
                value = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            else:
                raise RuntimeError(f"Unhandled meta buffer {name}; use a compatible pinned upstream revision")
            setattr(parent, attr, value)
    del state
    model = model.to(device).eval()
    images = load_and_preprocess_images([str(x) for x in paths], mode="max_size", image_resolution=args.resolution).to(device)
    logging.info("Running backbone on %s images", tuple(images.shape))
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
    arrays = {"depth": predictions["depth"][0, ..., 0].float().cpu().numpy(),
              "confidence": predictions["depth_conf"][0].float().cpu().numpy(),
              "extrinsics": extrinsics[0].float().cpu().numpy(),
              "intrinsics": intrinsics[0].float().cpu().numpy(),
              "images": images.permute(0, 2, 3, 1).float().cpu().numpy()}
    arrays["confidence"] = arrays["confidence"].reshape(arrays["depth"].shape)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError("Non-finite model predictions; no demo exported")
    np.savez_compressed(args.out / "predictions.npz", **arrays)
    points = np.stack([unproject(d, k, e) for d, k, e in zip(arrays["depth"], arrays["intrinsics"], arrays["extrinsics"])])
    valid = (arrays["depth"] > 0) & np.isfinite(points).all(-1)
    rgb = (arrays["images"] * 255).clip(0, 255).astype(np.uint8)
    before_filter = int(valid.sum())
    if args.filter_white_bg:
        valid &= ~(rgb > 240).all(axis=-1)
    if args.filter_black_bg:
        valid &= rgb.sum(axis=-1) >= 16
    logging.info("Background filtering retained %d / %d valid points", valid.sum(), before_filter)
    if not valid.any():
        raise RuntimeError("No valid points remain after background filtering; disable the filters and retry")
    threshold = np.quantile(arrays["confidence"][valid], .5)
    keep = valid & (arrays["confidence"] >= threshold)
    write_ply(args.out / "reconstruction.ply", points[keep], arrays["images"][keep])
    # Deterministic subsampling for a lightweight offline canvas viewer.
    candidates = np.flatnonzero(valid.ravel())
    indices = candidates[np.linspace(0, len(candidates) - 1, min(18000, len(candidates))).astype(int)]
    flat_points = points.reshape(-1, 3)[indices]
    center = np.median(points[keep], axis=0)
    scale = np.quantile(np.linalg.norm(points[keep] - center, axis=-1), .90)
    thumbs = []
    for rgb in arrays["images"]:
        buf = io.BytesIO()
        Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="JPEG")
        thumbs.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    conf = arrays["confidence"].ravel()[indices]
    ranks = np.searchsorted(np.sort(arrays["confidence"][valid]), conf) / valid.sum()
    payload = {"points": ((flat_points - center) / max(scale, 1e-8)).round(5).tolist(),
               "colors": (arrays["images"].reshape(-1, 3)[indices] * 255).astype(int).tolist(),
               "ranks": ranks.round(4).tolist(), "images": thumbs,
               "device": device, "resolution": args.resolution,
               "seconds": round(time.perf_counter() - start, 2)}
    (args.out / "demo.html").write_text(Path(__file__).with_name("real_viewer.html").read_text().replace("__RESULTS__", json.dumps(payload)))
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "images": [str(x) for x in paths],
                "device": device, "resolution": args.resolution, "seconds": payload["seconds"],
                "exported_points": int(keep.sum()), "confidence_percentile": 50,
                "filter_white_bg": args.filter_white_bg, "filter_black_bg": args.filter_black_bg,
                "upstream_revision": "a3ab0141f96838724423541044ff5ba301cfd36a"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("Saved %s", json.dumps(manifest))


if __name__ == "__main__":
    main()
