"""Sign classifier v2: input features are computed inside the network, so the exported model takes
normalised points straight from the extractor and the phone repeats only data_v2.normalise.

Backbone after the 1st-place Kaggle ISLR design (1D depthwise-conv blocks with channel attention,
interleaved with transformer blocks, over MediaPipe landmarks), sized for INT8 on a phone CPU.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_v2 as dv

NORM_POINT = 9  # middle-finger knuckle: wrist-to-knuckle length sets the hand's scale


class Features(nn.Module):
    """x: B x T x P x 3 points, w: B x T x 2 x 21 x 3 metric hands, s: B x T x 4 seen flags -> B x T x F."""

    def __init__(self, face=True, world=True, z=True):
        super().__init__()
        self.face, self.world, self.c = face, world, 3 if z else 2
        body = list(range(dv.NPOSE)) + list(range(dv.LH[0], dv.RH[1]))
        sel = body + (list(range(*dv.FACE)) if face else [])
        self.register_buffer("sel", torch.tensor(sel), persistent=False)
        self.register_buffer("body", torch.tensor(body), persistent=False)
        self.dim = len(sel) * self.c + 42 * self.c + len(body) * self.c + (126 + 18 if world else 0) + 4

    @staticmethod
    def palm_frame(w):
        """Wrist-relative metric hand points (... x 21 x 3) -> the same points in the hand's own palm frame,
        and that frame's axes. Handshape becomes independent of how the hand or the camera is turned."""
        def unit(v):
            return v / (v.norm(dim=-1, keepdim=True) + 1e-6)

        def cross(a, b):  # written out: exports to plain ONNX ops
            return torch.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                                a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)
        y = unit(w[..., 9, :])                                  # wrist -> middle-finger knuckle
        z = unit(cross(w[..., 5, :] - w[..., 17, :], y))        # palm normal
        x = cross(y, z)
        axes = torch.stack([x, y, z], -2)                       # ... x 3 x 3
        return w @ axes.transpose(-1, -2), axes

    def hand_local(self, x, lo, hi):
        h = x[:, :, lo:hi, :self.c]
        local = h - h[:, :, :1]
        size = local[:, :, NORM_POINT].norm(dim=-1, keepdim=True).unsqueeze(-1)
        return (local / size.clamp_min(0.05)).clamp(-dv.CLAMP, dv.CLAMP)

    def forward(self, x, w, s):
        b, t = x.shape[:2]
        pos = x.index_select(2, self.sel)[..., :self.c]
        body = x.index_select(2, self.body)[..., :self.c]
        vel = torch.cat([torch.zeros_like(body[:, :1]), body[:, 1:] - body[:, :-1]], 1)
        parts = [pos.reshape(b, t, -1),
                 self.hand_local(x, *dv.LH).reshape(b, t, -1),
                 self.hand_local(x, *dv.RH).reshape(b, t, -1),
                 (vel * 4.0).clamp(-dv.CLAMP, dv.CLAMP).reshape(b, t, -1)]
        if self.world:
            shape, axes = self.palm_frame(w)
            parts += [shape.clamp(-dv.CLAMP, dv.CLAMP).reshape(b, t, -1), axes.reshape(b, t, -1)]
        parts.append(s)
        return torch.cat(parts, -1)


class ECA(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):  # B x C x T
        g = self.conv(x.mean(2, keepdim=True).transpose(1, 2)).transpose(1, 2)
        return x * torch.sigmoid(g)


class ConvBlock(nn.Module):
    def __init__(self, d, k=17, expand=2, drop=0.2):
        super().__init__()
        e = d * expand
        self.expand = nn.Conv1d(d, e, 1)
        self.dw = nn.Conv1d(e, e, k, padding=k // 2, groups=e, bias=False)
        self.bn = nn.BatchNorm1d(e)
        self.eca = ECA()
        self.project = nn.Conv1d(e, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # B x C x T
        h = F.silu(self.expand(x))
        h = self.eca(F.silu(self.bn(self.dw(h))))
        return x + self.drop(self.project(h))


class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4, expand=2, drop=0.2):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.BatchNorm1d(d)
        self.qkv = nn.Conv1d(d, 3 * d, 1)
        self.proj = nn.Conv1d(d, d, 1)
        self.norm2 = nn.BatchNorm1d(d)
        self.ff1 = nn.Conv1d(d, d * expand, 1)
        self.ff2 = nn.Conv1d(d * expand, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # B x C x T
        b, c, t = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, 3, self.heads, c // self.heads, t).unbind(1)
        att = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c // self.heads), dim=-1)  # B x H x T x T
        h = (v @ att.transpose(-1, -2)).reshape(b, c, t)
        x = x + self.drop(self.proj(h))
        return x + self.drop(self.ff2(F.silu(self.ff1(self.norm2(x)))))


class CosineHead(nn.Module):
    """Logits from the cosine between the normalised embedding and each normalised class weight, times a
    learned scale: keeps rare classes from being out-voted by weight norm when there are many classes."""

    def __init__(self, d, classes, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(classes, d) * 0.01)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, x):
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)) * self.log_scale.exp()

    def folded(self):
        """The same function as a plain Linear on normalised input, so INT8 export quantizes it as a Gemm."""
        lin = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(F.normalize(self.weight, dim=-1) * self.log_scale.exp())
        return nn.Sequential(Normalize(), lin)


class Normalize(nn.Module):
    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


class SignNet(nn.Module):
    def __init__(self, classes, dim=192, repeats=2, convs=3, kernel=17, heads=4, drop=0.2, head_drop=0.4,
                 face=True, world=True, z=True, t_in=64, head="linear", pool="mean"):
        super().__init__()
        self.t_in = t_in
        # "centre": the time steps are averaged with a weight that peaks in the middle of the window (a raised
        # cosine over a floor of 0.25), for a live model whose window is centred on the sign it should name.
        wt = 0.25 + 0.75 * (0.5 - 0.5 * torch.cos(2 * math.pi * (torch.arange(t_in) + 0.5) / t_in))
        self.register_buffer("pool_w", (wt / wt.sum()) if pool == "centre" else torch.full((t_in,), 1.0 / t_in), persistent=False)
        self.pool = pool
        self.features = Features(face, world, z)
        self.stem = nn.Conv1d(self.features.dim, dim, 1, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim)
        blocks = []
        for _ in range(repeats):
            blocks += [ConvBlock(dim, kernel, 2, drop) for _ in range(convs)]
            blocks.append(TransformerBlock(dim, heads, 2, drop))
        self.blocks = nn.Sequential(*blocks)
        self.top = nn.Conv1d(dim, 2 * dim, 1)
        self.head_drop = nn.Dropout(head_drop)
        self.head = CosineHead(2 * dim, classes) if head == "cosine" else nn.Linear(2 * dim, classes)

    def embed(self, x, w, s):
        f = self.features(x, w, s).transpose(1, 2)
        h = self.blocks(self.stem_bn(self.stem(f)))
        h = F.silu(self.top(h))
        return h.mean(2) if self.pool == "mean" else (h * self.pool_w).sum(2)

    def forward(self, x, w, s):
        return self.head(self.head_drop(self.embed(x, w, s)))


def build(cfg, classes):
    keys = ("dim", "repeats", "convs", "kernel", "heads", "drop", "head_drop", "face", "world", "z", "t_in", "head", "pool")
    return SignNet(classes, **{k: cfg[k] for k in keys if k in cfg})
