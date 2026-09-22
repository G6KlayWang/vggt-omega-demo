"""An original, small implementation of the register bottleneck in VGGT-Omega.

Input: matched 2D landmarks and known orthographic camera directions, NOT RGB.
Paper §3.1.2 inspires the routing; no upstream model code is imported here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self, dim=48, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class RegisterStage(nn.Module):
    """Local gathering, inter-view exchange, then local redistribution."""
    def __init__(self, dim=48, registers=4, mode="register"):
        super().__init__()
        if mode not in {"register", "global", "local"}:
            raise ValueError(mode)
        self.registers, self.mode = registers, mode
        self.local = AttentionBlock(dim)
        self.exchange = AttentionBlock(dim)

    def forward(self, x, cut_exchange=False):
        b, views, tokens, dim = x.shape
        # Each view gathers its own evidence into its registers.
        x = self.local(x.reshape(b * views, tokens, dim)).reshape(b, views, tokens, dim)
        if self.mode == "local" or cut_exchange:
            return x
        if self.mode == "global":
            return self.exchange(x.reshape(b, views * tokens, dim)).reshape_as(x)
        # The key idea: ONLY registers cross the view boundary.
        r = self.registers
        scene = x[:, :, :r].reshape(b, views * r, dim)
        scene = self.exchange(scene).reshape(b, views, r, dim)
        return torch.cat((scene, x[:, :, r:]), dim=2)


class TinyOmega(nn.Module):
    def __init__(self, mode="register", points=16, dim=48, registers=4, stages=2):
        super().__init__()
        self.registers = registers
        self.input = nn.Linear(4, dim)  # observed u,v and camera sin(theta), cos(theta)
        self.landmark_id = nn.Parameter(torch.randn(1, 1, points, dim) * 0.1)
        self.scene = nn.Parameter(torch.randn(1, 1, registers, dim) * 0.1)
        self.stages = nn.ModuleList([RegisterStage(dim, registers, mode) for _ in range(stages)])
        self.redistribute = AttentionBlock(dim)
        self.depth = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, observations, cut_exchange=False):
        b, views, _, _ = observations.shape
        patches = self.input(observations) + self.landmark_id
        x = torch.cat((self.scene.expand(b, views, -1, -1), patches), dim=2)
        for stage in self.stages:
            x = stage(x, cut_exchange)
        # Cross-view information reaches reference patches only after local attention.
        x = self.redistribute(x[:, 0])
        return self.depth(x[:, self.registers:]).squeeze(-1)


def attention_pairs(views, points=16, registers=4, stages=2):
    """Exact score-entry counts for this toy (per head/sample), not FLOPs or memory."""
    t = points + registers
    local = stages * views * t * t + t * t  # last block processes reference only
    return {"local": local, "register": local + stages * (views * registers) ** 2,
            "global": local + stages * (views * t) ** 2}
