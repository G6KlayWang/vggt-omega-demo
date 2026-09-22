"""Original pinhole unprojection, independent of the upstream geometry helpers."""
import numpy as np


def unproject(depth, intrinsics, world_to_camera):
    """Camera z-depth HxW -> world points HxWx3; OpenCV camera convention."""
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    pixels = np.stack((u, v, np.ones_like(u)), -1)
    camera = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
    rotation, translation = world_to_camera[:3, :3], world_to_camera[:3, 3]
    return (camera - translation) @ rotation


def write_ply(path, points, colors):
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (*p, *c))
