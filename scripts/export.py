"""Export SignNet / SignNetV2 / DualStreamSignNet / ConvBiGRUNet checkpoints to ONNX (FP32 and INT8)
for ONNX Runtime on Android, check that INT8 keeps test accuracy, and time it on one CPU thread.

The exported graph takes the clip already resampled to t_in steps and does everything else itself:
dominant-hand flip (if the model was trained with it), features, network, and averaging over an ensemble.
Inputs:  points [1, T, 119, 3], world [1, T, 2, 21, 3], seen [1, T, 4]   Output: probs [1, classes]

Usage: export.py <index.json> <cache.pt> <out_dir> <ckpt.pt> [<ckpt.pt> ...] [--subset 100] [--calib 400]
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import model as mv
import normalise as dv
import train as tv


def _bn_params(bn: nn.BatchNorm1d):
    inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias - bn.running_mean * inv
    return inv, bias


def _conv1_to_linear(conv: nn.Conv1d) -> nn.Linear:
    lin = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
    with torch.no_grad():
        lin.weight.copy_(conv.weight.squeeze(-1))
        if conv.bias is not None:
            lin.bias.copy_(conv.bias)
    return lin


def _fold_conv1_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Linear:
    inv, b = _bn_params(bn)
    lin = nn.Linear(conv.in_channels, conv.out_channels, bias=True)
    with torch.no_grad():
        lin.weight.copy_(conv.weight.squeeze(-1) * inv[:, None])
        cb = conv.bias if conv.bias is not None else torch.zeros_like(b)
        lin.bias.copy_(cb * inv + b)
    return lin


def _fold_bn_conv1(bn: nn.BatchNorm1d, conv: nn.Conv1d) -> nn.Linear:
    inv, b = _bn_params(bn)
    lin = nn.Linear(conv.in_channels, conv.out_channels, bias=True)
    with torch.no_grad():
        w = conv.weight.squeeze(-1)
        lin.weight.copy_(w * inv[None, :])
        cb = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
        lin.bias.copy_(w @ b + cb)
    return lin


def _fold_dw_bn(dw: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    inv, b = _bn_params(bn)
    out = nn.Conv1d(dw.in_channels, dw.out_channels, dw.kernel_size[0],
                    padding=dw.padding[0], groups=dw.groups, bias=True)
    with torch.no_grad():
        out.weight.copy_(dw.weight * inv[:, None, None])
        cb = dw.bias if dw.bias is not None else torch.zeros_like(b)
        out.bias.copy_(cb * inv + b)
    return out


def _fold_rep_dw(orig: mv.RepConvBlock) -> nn.Conv1d:
    """Algebraically merge dw_long+bn_long, dw_short+bn_short, and bn_id into one single depthwise Conv1d."""
    e = orig.dw_long.in_channels
    k_long = orig.k_long
    k_short = orig.k_short
    out = nn.Conv1d(e, e, k_long, padding=k_long // 2, groups=e, bias=True)
    with torch.no_grad():
        inv_l, b_l = _bn_params(orig.bn_long)
        w = orig.dw_long.weight * inv_l[:, None, None]
        inv_s, b_s = _bn_params(orig.bn_short)
        ws = orig.dw_short.weight * inv_s[:, None, None]
        pad = (k_long - k_short) // 2
        w[:, :, pad:pad + k_short] += ws
        inv_i, b_i = _bn_params(orig.bn_id)
        w[:, 0, k_long // 2] += inv_i
        out.weight.copy_(w)
        out.bias.copy_(b_l + b_s + b_i)
    return out


class FastFeatures(nn.Module):
    """Exact same features as model.Features without redundant full-tensor Gather copies."""

    def __init__(self, orig: mv.Features):
        super().__init__()
        self.face = orig.face
        self.world = orig.world
        self.c = orig.c
        self.dim = orig.dim
        self.body_end = dv.RH[1]
        self.lh_lo, self.rh_hi = dv.LH[0], dv.RH[1]
        self.contiguous_all = self.face and self.c == 3

    def forward(self, x, w, s):
        b, t = x.shape[:2]
        if self.contiguous_all:
            pos = x
            body = x[:, :, :self.body_end, :]
        else:
            body = x[:, :, :self.body_end, :self.c]
            pos = x[:, :, :, :self.c] if self.face else body
        hands = x[:, :, self.lh_lo:self.rh_hi, :self.c].reshape(b, t, 2, 21, self.c)
        local = hands - hands[:, :, :, :1, :]
        size = local[:, :, :, mv.NORM_POINT, :].norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(0.05)
        hands_local = (local / size).clamp(-dv.CLAMP, dv.CLAMP).reshape(b, t, 42 * self.c)
        vel = torch.cat([torch.zeros_like(body[:, :1]), body[:, 1:] - body[:, :-1]], 1)
        vel = (vel * 4.0).clamp(-dv.CLAMP, dv.CLAMP).reshape(b, t, -1)
        parts = [pos.reshape(b, t, -1), hands_local, vel]
        if self.world:
            shape, axes = mv.Features.palm_frame(w)
            parts += [shape.clamp(-dv.CLAMP, dv.CLAMP).reshape(b, t, -1), axes.reshape(b, t, -1)]
        parts.append(s)
        return torch.cat(parts, -1)


class FastFeaturesV2(nn.Module):
    """Exact bit-equivalent vectorized FeaturesV2: replaces fragmented per-landmark gathers, stacks,
    and scalar norms with batched index_select + one [16, 15] anchor projection (removes ~178 ONNX nodes)."""

    def __init__(self, orig: mv.FeaturesV2):
        super().__init__()
        self.face, self.world, self.c = orig.face, orig.world, orig.c
        self.body_end = orig.body_end
        self.lh_lo, self.rh_hi = orig.lh_lo, orig.rh_hi
        self.dim, self.hand_dim, self.body_dim, self.contact_dim = orig.dim, orig.hand_dim, orig.body_dim, orig.contact_dim
        self.register_buffer("parents", orig.parents.clone(), persistent=False)
        self.register_buffer("key2_idx", orig.key2_idx.clone(), persistent=False)
        self.register_buffer("face_key_idx", orig.face_key_idx.clone(), persistent=False)
        ia = list(mv.TIP_PAIRS_A) + list(mv.HAND_TIPS) + list(mv.HAND_TIPS)
        ib = list(mv.TIP_PAIRS_B) + [0] * 5 + list(mv.HAND_MCPS)
        self.register_buffer("intra_a", torch.tensor(ia, dtype=torch.long), persistent=False)
        self.register_buffer("intra_b", torch.tensor(ib, dtype=torch.long), persistent=False)
        m_anc = torch.zeros(16, 15, dtype=torch.float32)
        for off in (0, 8):
            m_anc[off + 0, 0] = 1.0
            m_anc[off + 1, 1] = 0.5
            m_anc[off + 1, 2] = 0.5
            m_anc[off + 2, 5] = 0.5
            m_anc[off + 2, 6] = 0.5
            m_anc[off + 3, 7] = 0.5
            m_anc[off + 3, 8] = 0.5
            m_anc[off + 4, 13] = 0.5
            m_anc[off + 4, 14] = 0.5
        m_anc[5, 3] = 1.0
        m_anc[6, 7] = 1.0
        m_anc[7, 8] = 1.0
        m_anc[13, 4] = 1.0
        m_anc[14, 8] = 1.0
        m_anc[15, 7] = 1.0
        self.register_buffer("m_anc_t", m_anc.t().contiguous(), persistent=False)
        self.register_buffer("h2h_l", torch.tensor([0, 9, 4, 8, 12, 8, 9], dtype=torch.long), persistent=False)
        self.register_buffer("h2h_r", torch.tensor([0, 9, 4, 8, 12, 9, 8], dtype=torch.long), persistent=False)
        self.register_buffer("speed_idx", torch.tensor([11, 12, dv.LH[0] + 9, dv.RH[0] + 9], dtype=torch.long), persistent=False)
        self.register_buffer("h_anc_idx", torch.tensor([dv.LH[0], dv.LH[0] + 9, dv.RH[0], dv.RH[0] + 9], dtype=torch.long), persistent=False)
        self.register_buffer("hw_v_idx", torch.tensor([11, 12] + list(range(dv.LH[0], dv.RH[1])), dtype=torch.long), persistent=False)
        self.register_buffer("pi_idx", torch.tensor([9, 8], dtype=torch.long), persistent=False)
        self.register_buffer("c1", torch.tensor([1, 2, 0], dtype=torch.long), persistent=False)
        self.register_buffer("c2", torch.tensor([2, 0, 1], dtype=torch.long), persistent=False)

    def _cross(self, a, b):
        return a.index_select(-1, self.c1) * b.index_select(-1, self.c2) - a.index_select(-1, self.c2) * b.index_select(-1, self.c1)

    def _palm_frame(self, w):
        y = w[:, :, :, 9, :]
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
        v = w[:, :, :, 5, :] - w[:, :, :, 17, :]
        z = self._cross(v, y)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        x = self._cross(y, z)
        axes = torch.stack([x, y, z], -2)
        return w @ axes.transpose(-1, -2), axes

    def _compute(self, x, w, s):
        b, t = x.shape[:2]
        c = self.c
        body = x[:, :, :self.body_end, :] if c == 3 else x[:, :, :self.body_end, :c]
        pose = body[:, :, :dv.NPOSE, :]
        hands = body[:, :, self.lh_lo:self.rh_hi, :].reshape(b, t, 2, 21, c)
        local = hands - hands[:, :, :, :1, :]
        size = local[:, :, :, mv.NORM_POINT, :].norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(0.05)
        hands_local = (local / size).clamp(-dv.CLAMP, dv.CLAMP)
        vel1_raw = torch.cat([torch.zeros_like(body[:, :1]), body[:, 1:] - body[:, :-1]], 1)
        vel1 = (vel1_raw * 4.0).clamp(-dv.CLAMP, dv.CLAMP)
        pose_vel1 = vel1[:, :, :dv.NPOSE, :]
        hands_vel1 = vel1[:, :, dv.LH[0]:dv.RH[1], :]
        hw_v = vel1_raw.index_select(2, self.hw_v_idx)
        hw_acc = (torch.cat([torch.zeros_like(hw_v[:, :1]), hw_v[:, 1:] - hw_v[:, :-1]], 1) * 4.0).clamp(-dv.CLAMP, dv.CLAMP)
        wrist_acc = hw_acc[:, :, :2, :]
        hands_acc = hw_acc[:, :, 2:, :]
        k2 = x.index_select(2, self.key2_idx) if c == 3 else x.index_select(2, self.key2_idx)[..., :c]
        vel2 = (torch.cat([torch.zeros_like(k2[:, :2]), k2[:, 2:] - k2[:, :-2]], 1) * 2.0).clamp(-dv.CLAMP, dv.CLAMP)
        if self.world:
            shape, axes = self._palm_frame(w)
            shape = shape.clamp(-dv.CLAMP, dv.CLAMP)
            ref_h = shape
        else:
            shape, axes = None, None
            ref_h = hands_local
        bones = (ref_h[:, :, :, 1:21, :] - ref_h.index_select(3, self.parents)).clamp(-dv.CLAMP, dv.CLAMP)
        intra_dists = (ref_h.index_select(3, self.intra_a) - ref_h.index_select(3, self.intra_b)).norm(dim=-1).clamp(0.0, dv.CLAMP)
        pose3 = x[:, :, :dv.NPOSE, :] if c != 3 else pose
        anc = torch.matmul(pose3.transpose(-1, -2), self.m_anc_t).transpose(-1, -2).reshape(b, t, 2, 8, 3)
        h3 = hands if c == 3 else x[:, :, self.lh_lo:self.rh_hi, :].reshape(b, t, 2, 21, 3)
        h_pi = h3.index_select(3, self.pi_idx)
        d_pi_anc = (h_pi.unsqueeze(4) - anc.unsqueeze(3)).norm(dim=-1).clamp(0.0, dv.CLAMP)
        d_palm_anc = d_pi_anc[:, :, :, 0, :]
        d_idx_anc = d_pi_anc[:, :, :, 1, :]
        h_palm = h_pi[:, :, :, 0, :]
        vec_nose = (h_palm - anc[:, :, 0:1, 0, :]).clamp(-dv.CLAMP, dv.CLAMP)
        vec_mouth = (h_palm - anc[:, :, 0:1, 2, :]).clamp(-dv.CLAMP, dv.CLAMP)
        lh3, rh3 = h3[:, :, 0], h3[:, :, 1]
        h2h_pts = (lh3.index_select(2, self.h2h_l) - rh3.index_select(2, self.h2h_r)).norm(dim=-1).clamp(0.0, dv.CLAMP)
        contact_parts = [d_palm_anc.reshape(b, t, 16), d_idx_anc.reshape(b, t, 16),
                         vec_nose.reshape(b, t, 6), vec_mouth.reshape(b, t, 6)]
        if self.face:
            face_pos = x[:, :, dv.FACE[0]:dv.FACE[1], :] if c == 3 else x[:, :, dv.FACE[0]:dv.FACE[1], :c]
            fk = x.index_select(2, self.face_key_idx) if c == 3 else x.index_select(2, self.face_key_idx)[..., :c]
            fvel = (torch.cat([torch.zeros_like(fk[:, :1]), fk[:, 1:] - fk[:, :-1]], 1) * 4.0).clamp(-dv.CLAMP, dv.CLAMP)
            brow_mid = 0.5 * (x[:, :, dv.FACE[0] + 58, :] + x[:, :, dv.FACE[0] + 59, :])
            low_lip = x[:, :, dv.FACE[0] + 15, :]
            f_anc = torch.stack([brow_mid, low_lip], 2).unsqueeze(2).unsqueeze(3)
            f_seen = s[:, :, 3:4]
            d_pif = (h_pi.unsqueeze(4) - f_anc).norm(dim=-1).clamp(0.0, dv.CLAMP)
            d_pf = d_pif[:, :, :, 0, :].reshape(b, t, 4) * f_seen
            d_if = d_pif[:, :, :, 1, :].reshape(b, t, 4) * f_seen
            contact_parts += [d_pf, d_if]
        else:
            face_pos, fvel = None, None
        contact_parts.append(h2h_pts)
        if self.world:
            palm_dots = (axes[:, :, 0] * axes[:, :, 1]).sum(-1).clamp(-1.0, 1.0)
            contact_parts.append(palm_dots)
        contacts = torch.cat(contact_parts, -1)
        speeds = vel1.index_select(2, self.speed_idx).norm(dim=-1).clamp(0.0, dv.CLAMP)
        h_anc_pos = body.index_select(2, self.h_anc_idx)
        return (pose, face_pos, body, hands_local, vel1, pose_vel1, hands_vel1,
                shape, axes, bones, intra_dists, contacts, hw_acc, wrist_acc, hands_acc,
                vel2, fvel, speeds, h_anc_pos, s)

    def forward(self, x, w, s):
        b, t = x.shape[:2]
        (pose, face_pos, body, hands_local, vel1, pose_vel1, hands_vel1,
         shape, axes, bones, intra_dists, contacts, hw_acc, wrist_acc, hands_acc,
         vel2, fvel, speeds, h_anc_pos, seen) = self._compute(x, w, s)
        pos = x if (self.face and self.c == 3) else (x[:, :, :, :self.c] if self.face else body)
        parts = [pos.reshape(b, t, -1), hands_local.reshape(b, t, -1), vel1.reshape(b, t, -1)]
        if self.world:
            parts += [shape.reshape(b, t, -1), axes.reshape(b, t, -1)]
        parts += [bones.reshape(b, t, -1), intra_dists.reshape(b, t, -1), contacts,
                  hw_acc.reshape(b, t, -1), vel2.reshape(b, t, -1)]
        if self.face:
            parts.append(fvel.reshape(b, t, -1))
        parts += [speeds, seen]
        return torch.cat(parts, -1)


class FastConvBlock(nn.Module):
    """ConvBlock or RepConvBlock in B x T x C layout: pointwise convs become Linear (fused DynamicQuantizeMatMul
    in ORT), all depthwise + BatchNorm1d branches fold into ONE depthwise conv, and ECA runs without transposes."""

    def __init__(self, orig):
        super().__init__()
        self.expand = _conv1_to_linear(orig.expand)
        if isinstance(orig, mv.RepConvBlock):
            self.dw = _fold_rep_dw(orig)
        else:
            self.dw = _fold_dw_bn(orig.dw, orig.bn)
        self.eca = nn.Conv1d(1, 1, orig.eca.conv.kernel_size[0], padding=orig.eca.conv.padding[0], bias=False)
        with torch.no_grad():
            self.eca.weight.copy_(orig.eca.conv.weight)
        self.project = _conv1_to_linear(orig.project)

    def forward(self, x):  # B x T x C
        h = F.silu(self.expand(x)).transpose(1, 2)
        h = F.silu(self.dw(h)).transpose(1, 2)
        g = torch.sigmoid(self.eca(h.mean(1, keepdim=True)))
        return x + self.project(h * g)


class FastTransformerBlock(nn.Module):
    """TransformerBlock in B x T x C layout with norm1 folded into qkv and norm2 folded into ff1."""

    def __init__(self, orig: mv.TransformerBlock):
        super().__init__()
        self.heads = orig.heads
        self.head_dim = orig.proj.in_channels // orig.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = _fold_bn_conv1(orig.norm1, orig.qkv)
        self.proj = _conv1_to_linear(orig.proj)
        self.ff1 = _fold_bn_conv1(orig.norm2, orig.ff1)
        self.ff2 = _conv1_to_linear(orig.ff2)

    def forward(self, x):  # B x T x C
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        att = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        h = (att @ v).transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(h)
        return x + self.ff2(F.silu(self.ff1(x)))


def _fold_blocks(seq: nn.Sequential) -> nn.Sequential:
    out = []
    for blk in seq:
        if isinstance(blk, (mv.ConvBlock, mv.RepConvBlock)):
            out.append(FastConvBlock(blk))
        elif isinstance(blk, mv.TransformerBlock):
            out.append(FastTransformerBlock(blk))
        else:
            raise TypeError(f"Unsupported block type: {type(blk)}")
    return nn.Sequential(*out)


class FastSignNet(nn.Module):
    """Inference-folded SignNet / SignNetV2: exact FP32 match to PyTorch, eliminating all BatchNorm1d and
    RepConv multi-branch ops and converting all pointwise Conv1d(k=1) to Linear for ORT DynamicQuantizeMatMul."""

    def __init__(self, orig: mv.SignNet):
        super().__init__()
        self.t_in = orig.t_in
        self.pool = orig.pool
        self.feat_ver = getattr(orig, "feat_ver", 1)
        self.register_buffer("pool_w", orig.pool_w.clone().view(1, -1, 1), persistent=False)
        self.features = FastFeaturesV2(orig.features) if isinstance(orig.features, mv.FeaturesV2) else FastFeatures(orig.features)
        self.stem = _fold_conv1_bn(orig.stem, orig.stem_bn)
        self.blocks = _fold_blocks(orig.blocks)
        self.top = _conv1_to_linear(orig.top)
        self.attn_pool = orig.attn_pool
        self.head = orig.head

    def embed_from_feat(self, feat_btc):
        h = self.stem(feat_btc)
        h = self.blocks(h)
        h = F.silu(self.top(h))
        if self.attn_pool is not None:
            return self.attn_pool.forward_btc(h)
        return h.mean(1) if self.pool == "mean" else (h * self.pool_w).sum(1)

    def embed(self, x, w, s):
        return self.embed_from_feat(self.features(x, w, s))

    def forward(self, x, w, s):
        return self.head(self.embed(x, w, s))


class FastDualStreamSignNet(nn.Module):
    """Inference-folded DualStreamSignNet in B x T x C layout."""

    def __init__(self, orig: mv.DualStreamSignNet):
        super().__init__()
        self.t_in = orig.t_in
        self.pool = orig.pool
        self.feat_ver = 2
        self.register_buffer("pool_w", orig.pool_w.clone().view(1, -1, 1), persistent=False)
        self.features = FastFeaturesV2(orig.features) if isinstance(orig.features, mv.FeaturesV2) else orig.features
        self.hand_stem = _fold_conv1_bn(orig.hand_stem, orig.hand_bn)
        self.body_stem = _fold_conv1_bn(orig.body_stem, orig.body_bn)
        self.hand_blocks = _fold_blocks(orig.hand_blocks)
        self.body_blocks = _fold_blocks(orig.body_blocks)
        self.gate_b2h = _conv1_to_linear(orig.gate_b2h)
        self.gate_h2b = _conv1_to_linear(orig.gate_h2b)
        self.fuse = _fold_conv1_bn(orig.fuse, orig.fuse_bn)
        self.joint_blocks = _fold_blocks(orig.joint_blocks)
        self.top = _conv1_to_linear(orig.top)
        self.attn_pool = orig.attn_pool
        self.head = orig.head

    def embed_from_streams(self, hf, bf):
        hh = self.hand_blocks(self.hand_stem(hf))
        hb = self.body_blocks(self.body_stem(bf))
        hh_g = hh * torch.sigmoid(self.gate_b2h(hb))
        hb_g = hb * torch.sigmoid(self.gate_h2b(hh))
        h = F.silu(self.fuse(torch.cat([hh_g, hb_g], -1)))
        h = F.silu(self.top(self.joint_blocks(h)))
        if self.attn_pool is not None:
            return self.attn_pool.forward_btc(h)
        return h.mean(1) if self.pool == "mean" else (h * self.pool_w).sum(1)

    def embed(self, x, w, s):
        hf, bf = self.features.forward_streams(x, w, s)
        return self.embed_from_streams(hf, bf)

    def forward(self, x, w, s):
        return self.head(self.embed(x, w, s))


class FastConvBiGRUNet(nn.Module):
    """Inference-folded ConvBiGRUNet in B x T x C layout."""

    def __init__(self, orig: mv.ConvBiGRUNet):
        super().__init__()
        self.t_in = orig.t_in
        self.pool = orig.pool
        self.feat_ver = orig.feat_ver
        self.register_buffer("pool_w", orig.pool_w.clone().view(1, -1, 1), persistent=False)
        self.features = FastFeaturesV2(orig.features) if isinstance(orig.features, mv.FeaturesV2) else FastFeatures(orig.features)
        self.stem = _fold_conv1_bn(orig.stem, orig.stem_bn)
        self.conv_pre = _fold_blocks(orig.conv_pre)
        self.gru1 = orig.gru1
        self.gru1_proj = _fold_conv1_bn(orig.gru1_proj, orig.gru1_bn)
        self.conv_mid = FastConvBlock(orig.conv_mid)
        self.gru2 = orig.gru2
        self.gru2_bn_top = _fold_bn_conv1(orig.gru2_bn, orig.top) if False else None
        # Note: gru2_bn -> SiLU -> top, so keep affine scale+bias for gru2_bn before SiLU!
        inv, b = _bn_params(orig.gru2_bn)
        self.register_buffer("g2_scale", inv.view(1, 1, -1))
        self.register_buffer("g2_bias", b.view(1, 1, -1))
        self.top = _conv1_to_linear(orig.top)
        self.attn_pool = orig.attn_pool
        self.head = orig.head

    def embed_from_feat(self, feat_btc):
        h = self.conv_pre(self.stem(feat_btc))
        g1, _ = self.gru1(h)
        h = h + self.gru1_proj(g1)
        h = self.conv_mid(h)
        g2, _ = self.gru2(h)
        h = F.silu(self.top(F.silu(g2 * self.g2_scale + self.g2_bias)))
        if self.attn_pool is not None:
            return self.attn_pool.forward_btc(h)
        return h.mean(1) if self.pool == "mean" else (h * self.pool_w).sum(1)

    def embed(self, x, w, s):
        return self.embed_from_feat(self.features(x, w, s))

    def forward(self, x, w, s):
        return self.head(self.embed(x, w, s))


def fold_model(m: nn.Module) -> nn.Module:
    if isinstance(m, mv.SignNet):
        return FastSignNet(m).eval()
    if isinstance(m, mv.DualStreamSignNet):
        return FastDualStreamSignNet(m).eval()
    if isinstance(m, mv.ConvBiGRUNet):
        return FastConvBiGRUNet(m).eval()
    raise TypeError(f"Unsupported model type for folding: {type(m)}")


@torch.jit.script
def _cond_flip(left: torch.Tensor, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, perm: torch.Tensor):
    if bool(left[0].item()):
        xf = x.index_select(2, perm)
        xf = torch.cat([-xf[..., :1], xf[..., 1:]], -1)
        wf = torch.stack([w[:, :, 1], w[:, :, 0]], 2)
        wf = torch.cat([-wf[..., :1], wf[..., 1:]], -1)
        sf = torch.stack([s[..., 0], s[..., 2], s[..., 1], s[..., 3]], -1)
        return xf, wf, sf
    return x, w, s


class Deploy(nn.Module):
    """Outputs: probs (what predict() answers from: the same-sign answer rule applied when the model has one),
    set_probs (per gloss, the probability of its whole same-sign set: the confidence the captions use),
    embedding (the pooled features, for signs the user teaches on the device), raw_probs (the network's own
    probabilities, before the rule). A live model (trained with a "no sign" class, train.py --cont) also returns
    none [1, 1], the probability that the window is centred between two signs; its word outputs then sum to
    1 - none."""

    def __init__(self, models, cfgs, face_flip, rule=None, cond_flip=False, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.canon = [bool(c["canon"]) for c in cfgs]
        self.canon_rel = [bool(c.get("canon_rel", False)) for c in cfgs]
        self.register_buffer("perm", tv.flip_perm(face_flip), persistent=False)
        self.t_in = cfgs[0]["t_in"]
        self.rule = rule
        self.none = bool(cfgs[0].get("none_class"))
        self.cond_flip = cond_flip
        if weights is None:
            ws = [1.0 / len(models)] * len(models)
        else:
            s = sum(weights)
            ws = [w / s for w in weights]
        self.weights = ws
        # Share CondFlip across ensemble members that share the exact same (canon, canon_rel) setting
        self.shared_canon = (len(models) > 1 and all(c == self.canon[0] for c in self.canon)
                             and all(r == self.canon_rel[0] for r in self.canon_rel))
        # Share FeaturesV2 across ensemble members when both use FeaturesV2 on the same canonicalized input
        self.shared_feat_v2 = (
            self.shared_canon
            and all(isinstance(getattr(m, "features", None), (mv.FeaturesV2, FastFeaturesV2)) for m in models)
            and all(m.features.face == models[0].features.face
                    and m.features.world == models[0].features.world
                    and m.features.c == models[0].features.c for m in models)
        )
        # Check if all models have matching embedding dim so we can combine embeddings cleanly for Teach-a-Sign
        self.norm_combine_emb = len(models) > 1

    def _dom_left(self, x, s, rel: bool):
        mid = (x[:, :, dv.SHOULDERS[0], :2] + x[:, :, dv.SHOULDERS[1], :2]) * 0.5 if rel else 0
        scores = []
        for slot, wi in ((1, tv.POSE_WRIST[0]), (2, tv.POSE_WRIST[1])):
            wr = x[:, :, wi, :2] - mid
            path = (wr[:, 1:] - wr[:, :-1]).norm(dim=-1).sum(1)
            scores.append(s[..., slot].mean(1) + 2.0 * path)
        return scores[0] > scores[1]

    def flip(self, x, w, s):
        xf = x.index_select(2, self.perm)
        xf = torch.cat([-xf[..., :1], xf[..., 1:]], -1)
        wf = torch.stack([w[:, :, 1], w[:, :, 0]], 2)
        wf = torch.cat([-wf[..., :1], wf[..., 1:]], -1)
        sf = torch.stack([s[..., 0], s[..., 2], s[..., 1], s[..., 3]], -1)
        return xf, wf, sf

    def _apply_canon(self, x, w, s, rel: bool):
        left = self._dom_left(x, s, rel)
        if self.cond_flip:
            return _cond_flip(left, x, w, s, self.perm)
        l4 = left[:, None, None, None]
        xf, wf, sf = self.flip(x, w, s)
        return torch.where(l4, xf, x), torch.where(l4[..., None], wf, w), torch.where(l4[:, :, :, 0], sf, s)

    def forward(self, x, w, s):
        probs = 0
        embs = []
        if self.shared_canon:
            xs, ws, ss = self._apply_canon(x, w, s, self.canon_rel[0]) if self.canon[0] else (x, w, s)
            if self.shared_feat_v2:
                # Compute shared FeaturesV2 intermediate tensors ONCE for both flat and dual-stream models!
                fmod = self.models[0].features
                b, t = xs.shape[:2]
                (pose, face_pos, body, hands_local, vel1, pose_vel1, hands_vel1,
                 shape, axes, bones, intra_dists, contacts, hw_acc, wrist_acc, hands_acc,
                 vel2, fvel, speeds, h_anc_pos, seen) = fmod._compute(xs, ws, ss)
                flat_feat = None
                stream_feat = None
                for m, wt in zip(self.models, self.weights):
                    if isinstance(m, (FastDualStreamSignNet, mv.DualStreamSignNet)):
                        if stream_feat is None:
                            h_parts = [hands_local.reshape(b, t, -1)]
                            if fmod.world:
                                h_parts += [shape.reshape(b, t, -1), axes.reshape(b, t, -1)]
                            h_parts += [bones.reshape(b, t, -1), intra_dists.reshape(b, t, -1),
                                        hands_vel1.reshape(b, t, -1), hands_acc.reshape(b, t, -1), seen[:, :, 1:3]]
                            b_parts = [pose.reshape(b, t, -1)]
                            if fmod.face:
                                b_parts.append(face_pos.reshape(b, t, -1))
                            b_parts += [h_anc_pos.reshape(b, t, -1), pose_vel1.reshape(b, t, -1),
                                        wrist_acc.reshape(b, t, -1), vel2.reshape(b, t, -1)]
                            if fmod.face:
                                b_parts.append(fvel.reshape(b, t, -1))
                            b_parts.append(contacts)
                            if fmod.world:
                                b_parts.append(axes.reshape(b, t, -1))
                            b_parts += [speeds, seen]
                            stream_feat = (torch.cat(h_parts, -1), torch.cat(b_parts, -1))
                        e = m.embed_from_streams(stream_feat[0], stream_feat[1]) if isinstance(m, FastDualStreamSignNet) else m.embed(xs, ws, ss)
                    else:
                        if flat_feat is None:
                            pos = xs if (fmod.face and fmod.c == 3) else (xs[:, :, :, :fmod.c] if fmod.face else body)
                            parts = [pos.reshape(b, t, -1), hands_local.reshape(b, t, -1), vel1.reshape(b, t, -1)]
                            if fmod.world:
                                parts += [shape.reshape(b, t, -1), axes.reshape(b, t, -1)]
                            parts += [bones.reshape(b, t, -1), intra_dists.reshape(b, t, -1), contacts,
                                      hw_acc.reshape(b, t, -1), vel2.reshape(b, t, -1)]
                            if fmod.face:
                                parts.append(fvel.reshape(b, t, -1))
                            parts += [speeds, seen]
                            flat_feat = torch.cat(parts, -1)
                        e = m.embed_from_feat(flat_feat)
                    probs = probs + wt * torch.softmax(m.head(e), -1)
                    embs.append(e / (e.norm(dim=-1, keepdim=True) + 1e-12) * math.sqrt(wt) if self.norm_combine_emb else e)
            else:
                for m, wt in zip(self.models, self.weights):
                    e = m.embed(xs, ws, ss)
                    probs = probs + wt * torch.softmax(m.head(e), -1)
                    embs.append(e / (e.norm(dim=-1, keepdim=True) + 1e-12) * math.sqrt(wt) if self.norm_combine_emb else e)
        else:
            for m, canon, rel, wt in zip(self.models, self.canon, self.canon_rel, self.weights):
                xi, wi, si = self._apply_canon(x, w, s, rel) if canon else (x, w, s)
                e = m.embed(xi, wi, si)
                probs = probs + wt * torch.softmax(m.head(e), -1)
                embs.append(e / (e.norm(dim=-1, keepdim=True) + 1e-12) * math.sqrt(wt) if self.norm_combine_emb else e)
        emb = embs[0] if len(embs) == 1 else torch.cat(embs, -1)
        raw = probs
        none = ()
        if self.none:   # the last class is "no sign": it leaves the word outputs and comes out on its own
            raw, none = raw[:, :-1], (raw[:, -1:],)
        if self.rule is None:
            return (raw, raw, emb, raw) + none
        ruled, mass = self.rule(raw)
        return (ruled, mass, emb, raw) + none


def batches(data, ids, t_in):
    out = []
    for i in ids.tolist():
        sel = torch.tensor([i])
        x, w, s = tv.sample_time(data["X"][sel], data["W"][sel], data["S"][sel], data["L"][sel], t_in, False)
        out.append(dict(points=x.numpy().astype(np.float32), world=w.numpy().astype(np.float32), seen=s.numpy().astype(np.float32)))
    return out


def run_onnx(path, feeds, y, threads=1, reference=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    probs, times = [], []
    for f in feeds:
        t = time.perf_counter()
        probs.append(sess.run(None, f)[0])
        times.append(1000 * (time.perf_counter() - t))
    probs = torch.from_numpy(np.concatenate(probs))
    times = sorted(times[5:])
    sc = tv.score(probs, y)
    if reference is not None:  # label-free fidelity: how often the ONNX model's top-1 equals the PyTorch model's
        sc["agree_top1_with_torch"] = round(100 * float((probs.argmax(1) == reference.argmax(1)).float().mean()), 2)
    return sc, dict(p50_ms=round(times[len(times) // 2], 3), p95_ms=round(times[int(0.95 * len(times))], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--calib", type=int, default=400)
    ap.add_argument("--ops", default="", help="op types quantized to INT8 (default: Gemm,MatMul when folded+dynamic, else Conv,Gemm)")
    ap.add_argument("--calib-method", default="MinMax", choices=["MinMax", "Entropy", "Percentile"],
                    help="how activation ranges are set from the calibration clips")
    ap.add_argument("--act-type", default="QInt8", choices=["QInt8", "QUInt8"], help="activation integer type (static)")
    ap.add_argument("--dynamic", action="store_true",
                    help="weight-only INT8 (activations quantized on the fly) instead of static calibration")
    ap.add_argument("--exclude", default="", help="comma-separated substrings of node names kept in FP32 (e.g. stem,attn_pool)")
    ap.add_argument("--labels", default="", help="JSON label list for vocabularies beyond the WLASL index (union)")
    ap.add_argument("--set-rule", action="store_true",
                    help="put the same-sign answer rule (signsets.py, sets from train+val listings) inside the graph")
    ap.add_argument("--no-fold", action="store_true",
                    help="keep the unfused Conv1d+BatchNorm1d graph instead of folding into B x T x C Linear + FastSetRule + CondFlip")
    ap.add_argument("--weights", default="", help="comma-separated ensemble weights matching ckpts")
    a = ap.parse_args()
    fold = not a.no_fold
    ops = a.ops or ("Gemm,MatMul" if (fold and a.dynamic) else "Conv,Gemm")
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cks = [torch.load(c, weights_only=False) for c in a.ckpts]
    weights = [float(x) for x in a.weights.split(",")] if a.weights else None
    models = []
    for ck in cks:
        m = mv.build(ck["cfg"], ck["classes"])
        m.load_state_dict(ck["state"])
        if isinstance(m.head, mv.CosineHead):
            m.head = m.head.folded()   # a plain Linear on normalised input, so it is quantized as a Gemm
        m.eval()
        models.append(fold_model(m) if fold else m)
    entries = json.load(open(a.index, encoding="utf-8"))
    rule, sets_info = None, None
    if a.set_rule:
        import signsets
        n_idx, n_cls = min(a.subset, len(entries)), cks[0]["classes"] - (1 if cks[0]["cfg"].get("none_class") else 0)
        set_of, links = signsets.same_sign_sets(entries, n_idx)
        counts = signsets.wlasl_counts(entries, n_idx)
        # Classes outside the WLASL index (a union vocabulary) are sets of their own with a neutral weight.
        set_of = torch.cat([set_of, int(set_of.max()) + 1 + torch.arange(n_cls - n_idx)])
        counts = torch.cat([counts, torch.ones(n_cls - n_idx)])
        rule = signsets.FastSetRule(set_of, counts) if fold else signsets.SetRule(set_of, counts)
        sets_info = dict(links=links, sets_of_two_or_more=int((torch.bincount(set_of) > 1).sum()),
                         gamma=signsets.GAMMA, beta=signsets.BETA, set_of=set_of.tolist())
    deploy = Deploy(models, [ck["cfg"] for ck in cks], cks[0].get("face_flip", torch.arange(62)),
                    rule, cond_flip=fold, weights=weights).eval()
    t_in = deploy.t_in

    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, a.subset)
    data = {k: cache[k] for k in ("X", "W", "S", "L")}
    test_ids, y = rows["test"]
    feeds = batches(data, test_ids, t_in)
    g = torch.Generator().manual_seed(0)
    calib_ids = rows["train"][0][torch.randperm(len(rows["train"][0]), generator=g)[:a.calib]]
    calib = batches(data, calib_ids, t_in)

    with torch.no_grad():
        outs = [deploy(*(torch.from_numpy(f[k]) for k in ("points", "world", "seen"))) for f in feeds]
        torch_probs = torch.cat([o[0] for o in outs])
        torch_raw = torch.cat([o[3] for o in outs])
    fp32, raw, int8 = out / "echogesture_fp32.onnx", out / "echogesture_pre.onnx", out / "echogesture_int8.onnx"
    ex = tuple(torch.from_numpy(feeds[0][k]) for k in ("points", "world", "seen"))
    torch.onnx.export(deploy, ex, str(fp32), input_names=["points", "world", "seen"],
                      output_names=["probs", "set_probs", "embedding", "raw_probs"] + (["none"] if deploy.none else []),
                      opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(raw))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            return next(self.it, None)

    import onnx
    keep = [k for k in a.exclude.split(",") if k]
    exclude = [n.name for n in onnx.load(str(raw)).graph.node if any(k in n.name for k in keep)]
    if a.dynamic:
        quantize_dynamic(str(raw), str(int8), per_channel=True, weight_type=QuantType.QInt8,
                         op_types_to_quantize=ops.split(","), nodes_to_exclude=exclude)
    else:
        quantize_static(str(raw), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=getattr(QuantType, a.act_type), weight_type=QuantType.QInt8,
                        op_types_to_quantize=ops.split(","), calibrate_method=getattr(CalibrationMethod, a.calib_method),
                        nodes_to_exclude=exclude)
    raw.unlink()
    res = dict(ckpts=a.ckpts, weights=weights, t_in=t_in, test_clips=len(y), none_class=deploy.none,
               int8_scheme=dict(dynamic=a.dynamic, folded=fold, ops=ops, fp32_nodes=exclude),
               torch=tv.score(torch_probs, y))
    if sets_info is not None:
        res["torch_before_rule"] = tv.score(torch_raw, y)
        res["same_sign"] = {k: v for k, v in sets_info.items() if k != "set_of"}
        (out / "sets.json").write_text(json.dumps(sets_info["set_of"]), encoding="utf-8")
    for name, path in (("fp32", fp32), ("int8", int8)):
        sc, lat = run_onnx(path, feeds, y, reference=torch_probs)
        res[name] = dict(size_mb=round(path.stat().st_size / 1e6, 3), test=sc, one_thread=lat)
    labels = json.loads(pathlib.Path(a.labels).read_text(encoding="utf-8")) if a.labels else [e["gloss"].upper() for e in entries[:a.subset]]
    labels = labels[:cks[0]["classes"] - (1 if deploy.none else 0)]
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (out / "export.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
