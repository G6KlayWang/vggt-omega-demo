"""Train three tiny models fairly, evaluate unseen shapes, export an offline demo."""
import argparse
import json
import logging
from pathlib import Path
import time
import torch
from core import TinyOmega, attention_pairs
from toy_data import make_batch, triangulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Training device: %s", device)
    args.out.mkdir(parents=True, exist_ok=True)
    validation, truth = make_batch(512, torch.Generator().manual_seed(9001))
    validation, truth = validation.to(device), truth.to(device)
    target = truth[..., 2]
    models, metrics, curves = {}, {}, {}
    for mode in ("local", "register", "global"):
        torch.manual_seed(42)
        model = TinyOmega(mode).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
        data_rng = torch.Generator().manual_seed(1234)
        start, curve = time.perf_counter(), []
        model.train()
        for step in range(1, args.steps + 1):
            obs, xyz = make_batch(32, data_rng)
            obs, xyz = obs.to(device), xyz.to(device)
            loss = (model(obs) - xyz[..., 2]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                curve.append([step, loss.item()])
                logging.info("%s step=%d/%d train_mse=%.6f", mode, step, args.steps, loss.item())
        model.eval()
        with torch.inference_mode():
            pred = model(validation)
            metrics[mode] = {"rmse": (pred - target).square().mean().sqrt().item(),
                             "train_seconds": time.perf_counter() - start,
                             "parameters": sum(x.numel() for x in model.parameters())}
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "mode": mode, "steps": args.steps}, args.out / f"toy_{mode}.pt")
        models[mode], curves[mode] = model, curve
        logging.info("%s held-out RMSE %.5f", mode, metrics[mode]["rmse"])
    with torch.inference_mode():
        cut = models["register"](validation, cut_exchange=True)
        metrics["cut"] = {"rmse": (cut - target).square().mean().sqrt().item()}
        metrics["oracle"] = {"rmse": (triangulate(validation.cpu()).to(device) - target).square().mean().sqrt().item()}
        metrics["zero"] = {"rmse": target.square().mean().sqrt().item()}
        examples = []
        for noise in (0.0, 0.015, 0.06, 0.15):
            obs, xyz = make_batch(12, torch.Generator().manual_seed(2026), noise)
            predictions = {m: model(obs.to(device)).tolist() for m, model in models.items()}
            predictions["cut"] = models["register"](obs.to(device), cut_exchange=True).tolist()
            predictions["oracle"] = triangulate(obs).tolist()
            for i in range(12):
                examples.append({"scene": i, "noise": noise, "truth": xyz[i].tolist(),
                                 "observations": obs[i, :, :, :2].tolist(),
                                 "predictions": {m: x[i] for m, x in predictions.items()}})
    report = {"metrics": metrics, "curves": curves, "steps": args.steps,
              "validation_scenes": 512, "training_scenes_per_model": 32 * args.steps,
              "pairs": attention_pairs(3), "examples": examples,
              "seeds": {"model": 42, "train": 1234, "test": 9001, "demo": 2026},
              "torch_version": torch.__version__, "device": device}
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    template = Path(__file__).with_name("viewer.html").read_text()
    (args.out / "demo.html").write_text(template.replace("__RESULTS__", json.dumps(report)))
    logging.info("Ready: %s/demo.html\n%s", args.out, json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
