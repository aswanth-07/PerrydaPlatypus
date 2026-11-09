"""Clip normalisation and model input features. The NumPy and torch feature paths must stay identical."""
import numpy as np

POSE_KEEP = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
NP = len(POSE_KEEP)
LEFT = (NP, NP + 21)
RIGHT = (NP + 21, NP + 42)
POINTS = NP + 42
SHOULDERS = (POSE_KEEP.index(11), POSE_KEEP.index(12))
# Lateral pairs inside POSE_KEEP, swapped by the mirror augmentation.
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
FEATURES = POINTS * 3 + 42 * 3 + POINTS * 3 + 3
CLAMP = 6.0
NORM_POINT = 9  # middle-finger knuckle: wrist-to-knuckle length sets the hand's scale


def normalise_clip(kp, wh):
    """T x 67 x 3 with NaN for unseen points -> T x 57 x 3 body-relative points, T x 3 seen flags."""
    kp = kp.astype(np.float64).copy()
    w, h = float(wh[0]), float(wh[1])
    side = max(w, h)
    kp[..., 0] *= w / side
    kp[..., 1] *= h / side
    kp[..., 2] *= w / side
    pts = np.concatenate([kp[:, POSE_KEEP], kp[:, 25:]], axis=1)

    seen = np.stack([
        np.isfinite(pts[:, :NP, 0]).any(axis=1),
        np.isfinite(pts[:, LEFT[0]:LEFT[1], 0]).any(axis=1),
        np.isfinite(pts[:, RIGHT[0]:RIGHT[1], 0]).any(axis=1),
    ], axis=1).astype(np.float32)

    a, b = pts[:, SHOULDERS[0]], pts[:, SHOULDERS[1]]
    both = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    if both.any():
        centre = ((a[both] + b[both]) / 2).mean(axis=0)
        scale = np.median(np.linalg.norm((a[both] - b[both])[:, :2], axis=1))
    else:
        centre = np.nanmedian(pts.reshape(-1, 3), axis=0) if np.isfinite(pts).any() else np.zeros(3)
        scale = 0.25
    pts = (pts - centre) / max(float(scale), 1e-2)

    # Fill tracking gaps over time; points never seen stay at zero with their seen flag off.
    t = np.arange(len(pts))
    for p in range(POINTS):
        ok = np.isfinite(pts[:, p, 0])
        if ok.all():
            continue
        if not ok.any():
            pts[:, p] = 0.0
            continue
        for c in range(3):
            pts[:, p, c] = np.interp(t, t[ok], pts[ok, p, c])
    return np.clip(pts, -CLAMP, CLAMP).astype(np.float32), seen


def resample(x, length):
    """Linear resampling along time to `length` steps: T x ... -> length x ..."""
    pos = np.linspace(0, len(x) - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[lo] + w * x[hi]


def _hand_local_np(x, rng):
    hand = x[:, rng[0]:rng[1]]
    local = hand - hand[:, :1]
    size = np.linalg.norm(local[:, NORM_POINT], axis=-1, keepdims=True)[..., None]
    return np.clip(local / np.maximum(size, 0.05), -CLAMP, CLAMP)


def make_features_np(x, seen):
    """T x 57 x 3, T x 3 -> T x FEATURES."""
    t = len(x)
    vel = np.zeros_like(x)
    vel[1:] = x[1:] - x[:-1]
    return np.concatenate([
        x.reshape(t, -1),
        _hand_local_np(x, LEFT).reshape(t, -1),
        _hand_local_np(x, RIGHT).reshape(t, -1),
        np.clip(vel * 4.0, -CLAMP, CLAMP).reshape(t, -1),
        seen,
    ], axis=1).astype(np.float32)


def make_features_torch(x, seen):
    """B x T x 57 x 3, B x T x 3 -> B x T x FEATURES."""
    import torch

    def hand_local(rng):
        hand = x[:, :, rng[0]:rng[1]]
        local = hand - hand[:, :, :1]
        size = local[:, :, NORM_POINT].norm(dim=-1, keepdim=True).unsqueeze(-1)
        return (local / size.clamp_min(0.05)).clamp(-CLAMP, CLAMP)

    b, t = x.shape[:2]
    vel = torch.zeros_like(x)
    vel[:, 1:] = x[:, 1:] - x[:, :-1]
    return torch.cat([
        x.reshape(b, t, -1),
        hand_local(LEFT).reshape(b, t, -1),
        hand_local(RIGHT).reshape(b, t, -1),
        (vel * 4.0).clamp(-CLAMP, CLAMP).reshape(b, t, -1),
        seen,
    ], dim=-1)
