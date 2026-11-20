"""Sign classifier: input features are computed inside the network, so the exported model takes
normalised points straight from the extractor and the phone repeats only normalise.normalise.

Supports:
  - SignNet (arch="signnet", feat_ver=1): original 1D depthwise-conv + ECA + Transformer backbone.
  - SignNetV2 (arch="signnet_v2", feat_ver=2): ASL geometric FeaturesV2 + multi-scale reparameterized
    depthwise conv (k=17 + k=5 + id, algebraically folded into a single k=17 conv at export) + ECA +
    Transformer + DropPath + Multi-Head Temporal Attention Pooling.
  - DualStreamSignNet (arch="dual", feat_ver=2): dedicated Handshape stream + Body/Location/Face stream
    with bi-directional cross-stream gating, joint temporal refinement, and training-only auxiliary
    per-stream supervision (discarded at export).
  - ConvBiGRUNet (arch="bigru", feat_ver=2): multi-scale RepConv front-end + bidirectional GRU +
    Multi-Head Temporal Attention Pooling (recurrent temporal inductive bias).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import normalise as dv

NORM_POINT = 9  # middle-finger knuckle: wrist-to-knuckle length sets the hand's scale

# Hand kinematic tree parents for joints 1..20 (joint 0 is wrist)
HAND_PARENTS = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
HAND_TIPS = [4, 8, 12, 16, 20]
HAND_MCPS = [2, 5, 9, 13, 17]
TIP_PAIRS_A = [4, 4, 4, 4, 8, 8, 8, 12, 12, 16]
TIP_PAIRS_B = [8, 12, 16, 20, 12, 16, 20, 16, 20, 20]
# Key mirror-symmetric face points (indices inside FACE 0..61):
# outer lips (left 0, top 5, right 10, bottom 15), inner lips (top 25, bottom 34), brows (56, 58, 59, 61)
FACE_KEY_REL = [0, 5, 10, 15, 25, 34, 56, 58, 59, 61]


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


class FeaturesV2(nn.Module):
    """ASL-grounded geometric features on top of normalised landmarks (zero Normaliser change).

    Returns either the concatenated full feature tensor (B x T x F) via forward(), or cleanly
    factorized (hand_feat, body_feat) streams via forward_streams() for DualStreamSignNet.
    """

    def __init__(self, face=True, world=True, z=True):
        super().__init__()
        self.face, self.world, self.c = face, world, 3 if z else 2
        self.body_end = dv.RH[1]
        self.lh_lo, self.rh_hi = dv.LH[0], dv.RH[1]
        self.register_buffer("parents", torch.tensor(HAND_PARENTS, dtype=torch.long), persistent=False)
        self.register_buffer("tips", torch.tensor(HAND_TIPS, dtype=torch.long), persistent=False)
        self.register_buffer("mcps", torch.tensor(HAND_MCPS, dtype=torch.long), persistent=False)
        self.register_buffer("pair_a", torch.tensor(TIP_PAIRS_A, dtype=torch.long), persistent=False)
        self.register_buffer("pair_b", torch.tensor(TIP_PAIRS_B, dtype=torch.long), persistent=False)
        # 16 key trajectory points for smooth 2-step velocity: elbows(9,10), wrists(11,12),
        # LH palm(15+9) + tips(15+4,8,12,16,20), RH palm(36+9) + tips(36+4,8,12,16,20)
        key2 = [9, 10, 11, 12,
                dv.LH[0] + 9, dv.LH[0] + 4, dv.LH[0] + 8, dv.LH[0] + 12, dv.LH[0] + 16, dv.LH[0] + 20,
                dv.RH[0] + 9, dv.RH[0] + 4, dv.RH[0] + 8, dv.RH[0] + 12, dv.RH[0] + 16, dv.RH[0] + 20]
        self.register_buffer("key2_idx", torch.tensor(key2, dtype=torch.long), persistent=False)
        face_key = [dv.FACE[0] + i for i in FACE_KEY_REL]
        self.register_buffer("face_key_idx", torch.tensor(face_key, dtype=torch.long), persistent=False)

        c = self.c
        # Hand-stream components:
        #   hands_local(42*c) + palm_shape(126 if world) + palm_axes(18 if world) +
        #   bones(40*3 if world else 40*c) + intra_dists(40) +
        #   hands_vel1(42*c) + hands_accel(42*c) + hand_seen(2)
        bone_d = 120 if world else 40 * c
        self.hand_dim = 42 * c + (144 if world else 0) + bone_d + 40 + 42 * c + 42 * c + 2
        # Body/location/face/contact stream components:
        #   pose_pos(15*c) + face_pos(62*c if face) + hand_anchor_pos(4*c: L/R wrist+palm) +
        #   pose_vel1(15*c) + wrist_accel(2*c) + key_vel2(16*c) + face_vel(10*c if face) +
        #   contacts(32 body dists + 12 nose/mouth vecs + (8 face dists if face) + 7 h2h dists + (3 palm dots if world)) +
        #   palm_axes(18 if world) + speed(4) + seen(4)
        self.contact_dim = 32 + 12 + (8 if face else 0) + 7 + (3 if world else 0)
        self.body_dim = (15 * c + (62 * c if face else 0) + 4 * c +
                         15 * c + 2 * c + 16 * c + (10 * c if face else 0) +
                         self.contact_dim + (18 if world else 0) + 4 + 4)
        # Full flat dimension (no duplication of palm_axes/seen/hand_anchor_pos):
        #   pos((57 + (62 if face)) * c) + hands_local(42*c) + body_vel1(57*c) +
        #   (126 + 18 if world) + bones + intra_dists(40) + contact_dim +
        #   hw_accel(44*c) + key_vel2(16*c) + (10*c if face) + speed(4) + seen(4)
        n_pos = dv.RH[1] + (62 if face else 0)
        self.dim = (n_pos * c + 42 * c + 57 * c + (144 if world else 0) +
                    bone_d + 40 + self.contact_dim + 44 * c + 16 * c + (10 * c if face else 0) + 4 + 4)

    def _compute(self, x, w, s):
        b, t = x.shape[:2]
        c = self.c
        pose = x[:, :, :dv.NPOSE, :c]
        hands = x[:, :, self.lh_lo:self.rh_hi, :c].reshape(b, t, 2, 21, c)
        body = x[:, :, :self.body_end, :c]

        # Hand-local normalized coordinates
        local = hands - hands[:, :, :, :1, :]
        size = local[:, :, :, NORM_POINT, :].norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(0.05)
        hands_local = (local / size).clamp(-dv.CLAMP, dv.CLAMP)  # B x T x 2 x 21 x c

        # 1-step velocity on body (pose + hands)
        vel1_raw = torch.cat([torch.zeros_like(body[:, :1]), body[:, 1:] - body[:, :-1]], 1)
        vel1 = (vel1_raw * 4.0).clamp(-dv.CLAMP, dv.CLAMP)  # B x T x 57 x c
        pose_vel1 = vel1[:, :, :dv.NPOSE, :]
        hands_vel1 = vel1[:, :, dv.LH[0]:dv.RH[1], :]

        # 2nd-order acceleration on wrists (11,12) + LH + RH (44 pts)
        hw_v = torch.cat([vel1_raw[:, :, 11:13, :], vel1_raw[:, :, dv.LH[0]:dv.RH[1], :]], 2)
        hw_acc_raw = torch.cat([torch.zeros_like(hw_v[:, :1]), hw_v[:, 1:] - hw_v[:, :-1]], 1)
        hw_acc = (hw_acc_raw * 4.0).clamp(-dv.CLAMP, dv.CLAMP)  # B x T x 44 x c
        wrist_acc = hw_acc[:, :, :2, :]
        hands_acc = hw_acc[:, :, 2:, :]

        # Smooth 2-step velocity on 16 key arm/hand trajectory points
        k2 = x.index_select(2, self.key2_idx)[..., :c]
        vel2_raw = torch.cat([torch.zeros_like(k2[:, :2]), k2[:, 2:] - k2[:, :-2]], 1)
        vel2 = (vel2_raw * 2.0).clamp(-dv.CLAMP, dv.CLAMP)  # B x T x 16 x c

        # Palm-frame 3D hands + bones + intra-hand fingertip & curl distances
        if self.world:
            shape, axes = Features.palm_frame(w)
            shape = shape.clamp(-dv.CLAMP, dv.CLAMP)  # B x T x 2 x 21 x 3
            ref_h = shape
        else:
            shape, axes = None, None
            ref_h = hands_local

        bones = (ref_h[:, :, :, 1:21, :] - ref_h.index_select(3, self.parents)).clamp(-dv.CLAMP, dv.CLAMP)
        d_pairs = (ref_h.index_select(3, self.pair_a) - ref_h.index_select(3, self.pair_b)).norm(dim=-1)  # B x T x 2 x 10
        tips_p = ref_h.index_select(3, self.tips)
        mcps_p = ref_h.index_select(3, self.mcps)
        d_tip_wr = (tips_p - ref_h[:, :, :, :1, :]).norm(dim=-1)  # B x T x 2 x 5
        d_tip_mcp = (tips_p - mcps_p).norm(dim=-1)                # B x T x 2 x 5
        intra_dists = torch.cat([d_pairs, d_tip_wr, d_tip_mcp], -1).clamp(0.0, dv.CLAMP)  # B x T x 2 x 20

        # ASL Location & Contact Geometry (computed in full 3D pose space x)
        nose = x[:, :, 0, :]
        eyes_mid = 0.5 * (x[:, :, 1, :] + x[:, :, 2, :])
        mouth_mid = 0.5 * (x[:, :, 5, :] + x[:, :, 6, :])
        sh_mid = 0.5 * (x[:, :, 7, :] + x[:, :, 8, :])
        hip_mid = 0.5 * (x[:, :, 13, :] + x[:, :, 14, :])
        # Mirror-equivariant anchors for LH (side 0) and RH (side 1): [B, T, 2, 8, 3]
        anc_l = torch.stack([nose, eyes_mid, mouth_mid, sh_mid, hip_mid, x[:, :, 3, :], x[:, :, 7, :], x[:, :, 8, :]], 2)
        anc_r = torch.stack([nose, eyes_mid, mouth_mid, sh_mid, hip_mid, x[:, :, 4, :], x[:, :, 8, :], x[:, :, 7, :]], 2)
        anc = torch.stack([anc_l, anc_r], 2)  # B x T x 2 x 8 x 3

        h3 = x[:, :, self.lh_lo:self.rh_hi, :].reshape(b, t, 2, 21, 3)
        h_palm = h3[:, :, :, 9, :]   # B x T x 2 x 3
        h_idx = h3[:, :, :, 8, :]    # B x T x 2 x 3
        d_palm_anc = (h_palm.unsqueeze(3) - anc).norm(dim=-1).clamp(0.0, dv.CLAMP)  # B x T x 2 x 8
        d_idx_anc = (h_idx.unsqueeze(3) - anc).norm(dim=-1).clamp(0.0, dv.CLAMP)    # B x T x 2 x 8
        vec_nose = (h_palm - nose.unsqueeze(2)).clamp(-dv.CLAMP, dv.CLAMP)         # B x T x 2 x 3
        vec_mouth = (h_palm - mouth_mid.unsqueeze(2)).clamp(-dv.CLAMP, dv.CLAMP)   # B x T x 2 x 3

        # Hand-to-hand distances (mirror-invariant)
        lh3, rh3 = h3[:, :, 0], h3[:, :, 1]
        h2h_pts = torch.stack([
            (lh3[:, :, 0] - rh3[:, :, 0]).norm(dim=-1),
            (lh3[:, :, 9] - rh3[:, :, 9]).norm(dim=-1),
            (lh3[:, :, 4] - rh3[:, :, 4]).norm(dim=-1),
            (lh3[:, :, 8] - rh3[:, :, 8]).norm(dim=-1),
            (lh3[:, :, 12] - rh3[:, :, 12]).norm(dim=-1),
            (lh3[:, :, 8] - rh3[:, :, 9]).norm(dim=-1),
            (rh3[:, :, 8] - lh3[:, :, 9]).norm(dim=-1),
        ], -1).clamp(0.0, dv.CLAMP)  # B x T x 7

        contact_parts = [d_palm_anc.reshape(b, t, 16), d_idx_anc.reshape(b, t, 16),
                         vec_nose.reshape(b, t, 6), vec_mouth.reshape(b, t, 6)]
        if self.face:
            face_pos = x[:, :, dv.FACE[0]:dv.FACE[1], :c]
            fk = x.index_select(2, self.face_key_idx)[..., :c]
            fvel = (torch.cat([torch.zeros_like(fk[:, :1]), fk[:, 1:] - fk[:, :-1]], 1) * 4.0).clamp(-dv.CLAMP, dv.CLAMP)
            brow_mid = 0.5 * (x[:, :, dv.FACE[0] + 58, :] + x[:, :, dv.FACE[0] + 59, :])
            low_lip = x[:, :, dv.FACE[0] + 15, :]
            f_anc = torch.stack([brow_mid, low_lip], 2).unsqueeze(2)  # B x T x 1 x 2 x 3
            f_seen = s[:, :, 3:4]
            d_pf = (h_palm.unsqueeze(3) - f_anc).norm(dim=-1).clamp(0.0, dv.CLAMP).reshape(b, t, 4) * f_seen
            d_if = (h_idx.unsqueeze(3) - f_anc).norm(dim=-1).clamp(0.0, dv.CLAMP).reshape(b, t, 4) * f_seen
            contact_parts += [d_pf, d_if]
        else:
            face_pos, fvel = None, None
        contact_parts.append(h2h_pts)
        if self.world:
            palm_dots = (axes[:, :, 0] * axes[:, :, 1]).sum(-1).clamp(-1.0, 1.0)  # B x T x 3
            contact_parts.append(palm_dots)
        contacts = torch.cat(contact_parts, -1)

        # Speed scalars: L/R wrist and L/R palm
        sp_lw = vel1[:, :, 11, :].norm(dim=-1, keepdim=True)
        sp_rw = vel1[:, :, 12, :].norm(dim=-1, keepdim=True)
        sp_lp = vel1[:, :, dv.LH[0] + 9, :].norm(dim=-1, keepdim=True)
        sp_rp = vel1[:, :, dv.RH[0] + 9, :].norm(dim=-1, keepdim=True)
        speeds = torch.cat([sp_lw, sp_rw, sp_lp, sp_rp], -1).clamp(0.0, dv.CLAMP)

        # Hand anchor positions (L/R wrist + L/R palm) for body stream
        h_anc_pos = torch.stack([hands[:, :, 0, 0], hands[:, :, 0, 9], hands[:, :, 1, 0], hands[:, :, 1, 9]], 2)

        return (pose, face_pos, body, hands_local, vel1, pose_vel1, hands_vel1,
                shape, axes, bones, intra_dists, contacts, hw_acc, wrist_acc, hands_acc,
                vel2, fvel, speeds, h_anc_pos, s)

    def forward(self, x, w, s):
        b, t = x.shape[:2]
        (pose, face_pos, body, hands_local, vel1, _pv1, _hv1,
         shape, axes, bones, intra_dists, contacts, hw_acc, _wa, _ha,
         vel2, fvel, speeds, _hap, seen) = self._compute(x, w, s)
        pos = x[:, :, :, :self.c] if self.face else body
        parts = [pos.reshape(b, t, -1),
                 hands_local.reshape(b, t, -1),
                 vel1.reshape(b, t, -1)]
        if self.world:
            parts += [shape.reshape(b, t, -1), axes.reshape(b, t, -1)]
        parts += [bones.reshape(b, t, -1), intra_dists.reshape(b, t, -1), contacts,
                  hw_acc.reshape(b, t, -1), vel2.reshape(b, t, -1)]
        if self.face:
            parts.append(fvel.reshape(b, t, -1))
        parts += [speeds, seen]
        return torch.cat(parts, -1)

    def forward_streams(self, x, w, s):
        b, t = x.shape[:2]
        (pose, face_pos, _body, hands_local, _v1, pose_vel1, hands_vel1,
         shape, axes, bones, intra_dists, contacts, _hwa, wrist_acc, hands_acc,
         vel2, fvel, speeds, h_anc_pos, seen) = self._compute(x, w, s)
        h_parts = [hands_local.reshape(b, t, -1)]
        if self.world:
            h_parts += [shape.reshape(b, t, -1), axes.reshape(b, t, -1)]
        h_parts += [bones.reshape(b, t, -1), intra_dists.reshape(b, t, -1),
                    hands_vel1.reshape(b, t, -1), hands_acc.reshape(b, t, -1),
                    seen[:, :, 1:3]]
        b_parts = [pose.reshape(b, t, -1)]
        if self.face:
            b_parts.append(face_pos.reshape(b, t, -1))
        b_parts += [h_anc_pos.reshape(b, t, -1), pose_vel1.reshape(b, t, -1),
                    wrist_acc.reshape(b, t, -1), vel2.reshape(b, t, -1)]
        if self.face:
            b_parts.append(fvel.reshape(b, t, -1))
        b_parts.append(contacts)
        if self.world:
            b_parts.append(axes.reshape(b, t, -1))
        b_parts += [speeds, seen]
        return torch.cat(h_parts, -1), torch.cat(b_parts, -1)


class DropPath(nn.Module):
    """Stochastic depth per sample (identity at eval/export)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


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


class RepConvBlock(nn.Module):
    """Multi-scale reparameterized depthwise ConvBlock (k_long + k_short + identity BN branches).
    At export, all three depthwise+BN branches fold algebraically into a single depthwise Conv1d(k_long),
    giving multi-scale temporal features with zero extra inference parameters or FLOPs."""

    def __init__(self, d, k=17, k_short=5, expand=2, drop=0.2, drop_path=0.0):
        super().__init__()
        e = d * expand
        self.k_long = k
        self.k_short = k_short
        self.expand = nn.Conv1d(d, e, 1)
        self.dw_long = nn.Conv1d(e, e, k, padding=k // 2, groups=e, bias=False)
        self.bn_long = nn.BatchNorm1d(e)
        self.dw_short = nn.Conv1d(e, e, k_short, padding=k_short // 2, groups=e, bias=False)
        self.bn_short = nn.BatchNorm1d(e)
        self.bn_id = nn.BatchNorm1d(e)
        self.eca = ECA()
        self.project = nn.Conv1d(e, d, 1)
        self.drop = nn.Dropout(drop)
        self.dp = DropPath(drop_path)

    def forward(self, x):  # B x C x T
        h = F.silu(self.expand(x))
        y = self.bn_long(self.dw_long(h)) + self.bn_short(self.dw_short(h)) + self.bn_id(h)
        y = self.eca(F.silu(y))
        return x + self.dp(self.drop(self.project(y)))


class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4, expand=2, drop=0.2, drop_path=0.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.BatchNorm1d(d)
        self.qkv = nn.Conv1d(d, 3 * d, 1)
        self.proj = nn.Conv1d(d, d, 1)
        self.norm2 = nn.BatchNorm1d(d)
        self.ff1 = nn.Conv1d(d, d * expand, 1)
        self.ff2 = nn.Conv1d(d * expand, d, 1)
        self.drop = nn.Dropout(drop)
        self.dp = DropPath(drop_path)

    def forward(self, x):  # B x C x T
        b, c, t = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, 3, self.heads, c // self.heads, t).unbind(1)
        att = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c // self.heads), dim=-1)  # B x H x T x T
        h = (v @ att.transpose(-1, -2)).reshape(b, c, t)
        x = x + self.dp(self.drop(self.proj(h)))
        return x + self.dp(self.drop(self.ff2(F.silu(self.ff1(self.norm2(x))))))


class AttnMeanPool(nn.Module):
    """4-head grouped temporal attention pooling with mean-pool residual.
    Accepts B x C x T (training) or B x T x C (folded export via forward_btc).
    With centre_prior=True, biases attention logits by log(raised-cosine window) for live models."""

    def __init__(self, d, heads=4, t_in=64, centre_prior=False):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.hd = d // heads
        self.score = nn.Linear(d, heads)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.centre_prior = centre_prior
        wt = 0.25 + 0.75 * (0.5 - 0.5 * torch.cos(2 * math.pi * (torch.arange(t_in) + 0.5) / t_in))
        self.register_buffer("pool_w", wt / wt.sum(), persistent=False)
        self.register_buffer("log_prior", torch.log(wt.clamp_min(1e-4)).view(1, t_in, 1), persistent=False)

    def forward_btc(self, h):  # B x T x C
        b, t, c = h.shape
        logits = self.score(h)  # B x T x H
        if self.centre_prior:
            logits = logits + self.log_prior
        w = torch.softmax(logits, dim=1)  # B x T x H
        attn = (h.reshape(b, t, self.heads, self.hd) * w.unsqueeze(-1)).sum(1).reshape(b, c)
        base = (h * self.pool_w.view(1, t, 1)).sum(1) if self.centre_prior else h.mean(1)
        return 0.5 * base + 0.5 * attn

    def forward(self, h):  # B x C x T
        return self.forward_btc(h.transpose(1, 2))


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
                 face=True, world=True, z=True, t_in=64, head="linear", pool="mean",
                 feat_ver=1, rep_conv=False, drop_path=0.0):
        super().__init__()
        self.t_in = t_in
        self.feat_ver = feat_ver
        self.rep_conv = rep_conv
        wt = 0.25 + 0.75 * (0.5 - 0.5 * torch.cos(2 * math.pi * (torch.arange(t_in) + 0.5) / t_in))
        self.register_buffer("pool_w", (wt / wt.sum()) if pool == "centre" else torch.full((t_in,), 1.0 / t_in), persistent=False)
        self.pool = pool
        self.features = FeaturesV2(face, world, z) if feat_ver == 2 else Features(face, world, z)
        self.stem = nn.Conv1d(self.features.dim, dim, 1, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim)
        total_blks = repeats * (convs + 1)
        dp_rates = [drop_path * i / max(total_blks - 1, 1) for i in range(total_blks)]
        blocks = []
        idx = 0
        for _ in range(repeats):
            for _ in range(convs):
                if rep_conv:
                    blocks.append(RepConvBlock(dim, kernel, 5, 2, drop, dp_rates[idx]))
                else:
                    blocks.append(ConvBlock(dim, kernel, 2, drop))
                idx += 1
            blocks.append(TransformerBlock(dim, heads, 2, drop, dp_rates[idx]))
            idx += 1
        self.blocks = nn.Sequential(*blocks)
        self.top = nn.Conv1d(dim, 2 * dim, 1)
        self.attn_pool = AttnMeanPool(2 * dim, 4, t_in, centre_prior=(pool == "centre_attn")) if pool in ("attn", "centre_attn") else None
        self.head_drop = nn.Dropout(head_drop)
        self.head = CosineHead(2 * dim, classes) if head == "cosine" else nn.Linear(2 * dim, classes)

    def embed_from_feat(self, feat_btc):
        f = feat_btc.transpose(1, 2)
        h = self.blocks(self.stem_bn(self.stem(f)))
        h = F.silu(self.top(h))
        if self.attn_pool is not None:
            return self.attn_pool(h)
        return h.mean(2) if self.pool == "mean" else (h * self.pool_w).sum(2)

    def embed(self, x, w, s):
        return self.embed_from_feat(self.features(x, w, s))

    def forward(self, x, w, s):
        return self.head(self.head_drop(self.embed(x, w, s)))


class DualStreamSignNet(nn.Module):
    """Two-stream ASL architecture:
      - Hand Stream (dh): fine 3D palm-frame handshape, finger bones, fingertip distances, hand dynamics.
      - Body/Location/Face Stream (db): macro pose, face/lips, hand-to-body/face/hand contact geometry.
      - Bi-directional Cross-Stream Gated Fusion -> Joint Conv-Transformer refinement -> AttnMeanPool.
      - Optional training-only auxiliary per-stream classifiers (aux_hand, aux_body) for regularization."""

    def __init__(self, classes, dim=192, kernel=17, heads=4, drop=0.2, head_drop=0.4,
                 face=True, world=True, z=True, t_in=64, head="linear", pool="attn",
                 drop_path=0.15, stream_dim=128):
        super().__init__()
        self.t_in = t_in
        self.dim = dim
        self.stream_dim = stream_dim
        self.pool = pool
        self.feat_ver = 2
        wt = 0.25 + 0.75 * (0.5 - 0.5 * torch.cos(2 * math.pi * (torch.arange(t_in) + 0.5) / t_in))
        self.register_buffer("pool_w", (wt / wt.sum()) if pool == "centre" else torch.full((t_in,), 1.0 / t_in), persistent=False)
        self.features = FeaturesV2(face, world, z)
        sd = stream_dim
        self.hand_stem = nn.Conv1d(self.features.hand_dim, sd, 1, bias=False)
        self.hand_bn = nn.BatchNorm1d(sd)
        self.body_stem = nn.Conv1d(self.features.body_dim, sd, 1, bias=False)
        self.body_bn = nn.BatchNorm1d(sd)

        self.hand_blocks = nn.Sequential(
            RepConvBlock(sd, kernel, 5, 2, drop, drop_path * 0.2),
            RepConvBlock(sd, kernel, 5, 2, drop, drop_path * 0.4),
            TransformerBlock(sd, heads, 2, drop, drop_path * 0.5),
        )
        self.body_blocks = nn.Sequential(
            RepConvBlock(sd, kernel, 5, 2, drop, drop_path * 0.2),
            RepConvBlock(sd, kernel, 5, 2, drop, drop_path * 0.4),
            TransformerBlock(sd, heads, 2, drop, drop_path * 0.5),
        )
        # Bi-directional cross-stream gating
        self.gate_b2h = nn.Conv1d(sd, sd, 1)
        self.gate_h2b = nn.Conv1d(sd, sd, 1)
        self.fuse = nn.Conv1d(2 * sd, dim, 1, bias=False)
        self.fuse_bn = nn.BatchNorm1d(dim)
        self.joint_blocks = nn.Sequential(
            RepConvBlock(dim, kernel, 5, 2, drop, drop_path * 0.7),
            RepConvBlock(dim, kernel, 5, 2, drop, drop_path * 0.85),
            TransformerBlock(dim, heads, 2, drop, drop_path),
        )
        self.top = nn.Conv1d(dim, 2 * dim, 1)
        self.attn_pool = AttnMeanPool(2 * dim, 4, t_in, centre_prior=(pool == "centre_attn")) if pool in ("attn", "centre_attn") else None
        self.head_drop = nn.Dropout(head_drop)
        self.head = CosineHead(2 * dim, classes) if head == "cosine" else nn.Linear(2 * dim, classes)

    def _pool(self, h):
        if self.attn_pool is not None:
            return self.attn_pool(h)
        return h.mean(2) if self.pool == "mean" else (h * self.pool_w).sum(2)

    def embed_streams(self, x, w, s):
        hf, bf = self.features.forward_streams(x, w, s)
        hh = self.hand_blocks(self.hand_bn(self.hand_stem(hf.transpose(1, 2))))
        hb = self.body_blocks(self.body_bn(self.body_stem(bf.transpose(1, 2))))
        hh_g = hh * torch.sigmoid(self.gate_b2h(hb))
        hb_g = hb * torch.sigmoid(self.gate_h2b(hh))
        h = F.silu(self.fuse_bn(self.fuse(torch.cat([hh_g, hb_g], 1))))
        h = F.silu(self.top(self.joint_blocks(h)))
        return self._pool(h), hh.mean(2), hb.mean(2)

    def embed(self, x, w, s):
        e, _, _ = self.embed_streams(x, w, s)
        return e

    def forward(self, x, w, s):
        return self.head(self.head_drop(self.embed(x, w, s)))


class ConvBiGRUNet(nn.Module):
    """Recurrent hybrid architecture: Multi-Scale RepConv stem + 2-layer BiGRU + ECA + AttnMeanPool.
    Provides a genuinely different (recurrent Markovian) inductive bias from self-attention."""

    def __init__(self, classes, dim=192, kernel=17, drop=0.2, head_drop=0.4,
                 face=True, world=True, z=True, t_in=64, head="linear", pool="attn",
                 feat_ver=2, drop_path=0.15):
        super().__init__()
        self.t_in = t_in
        self.dim = dim
        self.pool = pool
        self.feat_ver = feat_ver
        wt = 0.25 + 0.75 * (0.5 - 0.5 * torch.cos(2 * math.pi * (torch.arange(t_in) + 0.5) / t_in))
        self.register_buffer("pool_w", (wt / wt.sum()) if pool == "centre" else torch.full((t_in,), 1.0 / t_in), persistent=False)
        self.features = FeaturesV2(face, world, z) if feat_ver == 2 else Features(face, world, z)
        self.stem = nn.Conv1d(self.features.dim, dim, 1, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim)
        self.conv_pre = nn.Sequential(
            RepConvBlock(dim, kernel, 5, 2, drop, drop_path * 0.25),
            RepConvBlock(dim, kernel, 5, 2, drop, drop_path * 0.5),
        )
        self.gru1 = nn.GRU(dim, dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gru1_proj = nn.Conv1d(2 * dim, dim, 1)
        self.gru1_bn = nn.BatchNorm1d(dim)
        self.conv_mid = RepConvBlock(dim, kernel, 5, 2, drop, drop_path * 0.75)
        self.gru2 = nn.GRU(dim, dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2_bn = nn.BatchNorm1d(2 * dim)
        self.top = nn.Conv1d(2 * dim, 2 * dim, 1)
        self.attn_pool = AttnMeanPool(2 * dim, 4, t_in, centre_prior=(pool == "centre_attn")) if pool in ("attn", "centre_attn") else None
        self.drop = nn.Dropout(drop)
        self.head_drop = nn.Dropout(head_drop)
        self.head = CosineHead(2 * dim, classes) if head == "cosine" else nn.Linear(2 * dim, classes)

    def embed_from_feat(self, feat_btc):
        h = self.conv_pre(self.stem_bn(self.stem(feat_btc.transpose(1, 2))))
        g1, _ = self.gru1(h.transpose(1, 2))
        h = h + self.drop(self.gru1_bn(self.gru1_proj(g1.transpose(1, 2))))
        h = self.conv_mid(h)
        g2, _ = self.gru2(h.transpose(1, 2))
        h = F.silu(self.top(F.silu(self.gru2_bn(g2.transpose(1, 2)))))
        if self.attn_pool is not None:
            return self.attn_pool(h)
        return h.mean(2) if self.pool == "mean" else (h * self.pool_w).sum(2)

    def embed(self, x, w, s):
        return self.embed_from_feat(self.features(x, w, s))

    def forward(self, x, w, s):
        return self.head(self.head_drop(self.embed(x, w, s)))


def build(cfg, classes):
    arch = cfg.get("arch", "signnet")
    if arch == "dual":
        keys = ("dim", "kernel", "heads", "drop", "head_drop", "face", "world", "z", "t_in", "head", "pool", "drop_path", "stream_dim")
        return DualStreamSignNet(classes, **{k: cfg[k] for k in keys if k in cfg})
    if arch == "bigru":
        keys = ("dim", "kernel", "drop", "head_drop", "face", "world", "z", "t_in", "head", "pool", "feat_ver", "drop_path")
        return ConvBiGRUNet(classes, **{k: cfg[k] for k in keys if k in cfg})
    if arch == "signnet_v2":
        keys = ("dim", "repeats", "convs", "kernel", "heads", "drop", "head_drop", "face", "world", "z", "t_in", "head", "pool", "drop_path")
        kw = {k: cfg[k] for k in keys if k in cfg}
        kw["feat_ver"] = cfg.get("feat_ver", 2)
        kw["rep_conv"] = cfg.get("rep_conv", True)
        return SignNet(classes, **kw)
    keys = ("dim", "repeats", "convs", "kernel", "heads", "drop", "head_drop", "face", "world", "z", "t_in", "head", "pool", "feat_ver", "rep_conv", "drop_path")
    return SignNet(classes, **{k: cfg[k] for k in keys if k in cfg})
