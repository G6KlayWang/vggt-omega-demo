"""Procedural 4x4 reliefs with independent hidden depth coefficients."""
import math
import torch


def make_batch(count, generator, noise=0.015):
    grid = torch.linspace(-0.8, 0.8, 4)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    x, y = x.flatten(), y.flatten()
    basis = torch.stack((torch.ones_like(x), x, y, torch.sin(math.pi * x) * torch.cos(math.pi * y)), -1)
    coefficients = torch.rand(count, 4, generator=generator) * 1.4 - 0.7
    z = coefficients @ basis.T
    xyz = torch.stack((x.expand_as(z), y.expand_as(z), z), -1)
    # Frontal view contains exactly zero information about these random depths.
    theta = torch.tensor([0.0, -math.pi / 4, math.pi / 4])
    s, c = theta.sin(), theta.cos()
    u = x[None, None, :] * c[None, :, None] + z[:, None, :] * s[None, :, None]
    v = y[None, None, :].expand_as(u)
    uv = torch.stack((u, v), -1)
    uv = uv + torch.randn(uv.shape, generator=generator) * noise
    cameras = torch.stack((s, c), -1)[None, :, None, :].expand(count, -1, 16, -1)
    return torch.cat((uv, cameras), -1), xyz


def triangulate(observations):
    """Least-squares orthographic oracle; known correspondences and cameras."""
    a = observations[..., [3, 2]]  # u = cos(theta)*x + sin(theta)*z
    a = a.permute(0, 2, 1, 3)
    u = observations[..., 0].permute(0, 2, 1).unsqueeze(-1)
    return torch.linalg.lstsq(a, u).solution[..., 1, 0]
