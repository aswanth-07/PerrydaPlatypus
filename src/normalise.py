"""One point layout for both extractors, normalised once and cached as GPU-ready tensors.

Layout per frame (P = 119 points, x y z):
  0-14    pose: nose, eyes, ears, mouth corners, shoulders, elbows, wrists, hips (MediaPipe pose ids POSE_KEEP)
  15-35   person's left hand
  36-56   person's right hand
  57-118  face: 40 lips, 4 nose, 18 eye and brow points (Holistic only; zero with its seen flag off otherwise)
plus metric 3D hand points (2 x 21 x 3, Holistic only) and four seen flags (pose, left hand, right hand, face).

Normalisation, the part the phone must repeat exactly: make x, y, z isotropic using the frame's aspect,
centre on the clip's mean shoulder midpoint, divide by the clip's median shoulder width, fill tracking
gaps by linear interpolation in time per group, clamp to +-CLAMP.

Usage: normalise.py <kp_dir> <out.pt> [--format holistic|legacy]
"""
import argparse
import pathlib
import sys

import numpy as np

POSE_KEEP = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
NPOSE = len(POSE_KEEP)
LH = (NPOSE, NPOSE + 21)
RH = (NPOSE + 21, NPOSE + 42)
FACE = (NPOSE + 42, NPOSE + 42 + 62)
P = FACE[1]
T_MAX = 96
CLAMP = 6.0
SHOULDERS = (POSE_KEEP.index(11), POSE_KEEP.index(12))
POSE_FLIP = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
WORLD_SCALE = 10.0  # metres to roughly unit range for hands (~0.1 m across)


def fill_group(a, ok):
    """Linear interpolation over time for a group seen as a whole: a is T x n x 3, ok is T bools."""
    t = len(a)
    if ok.all():
        return a
    if not ok.any():
        return np.zeros_like(a)
    idx = np.arange(t)
    prev = np.maximum.accumulate(np.where(ok, idx, -1))
    nxt = np.minimum.accumulate(np.where(ok, idx, t)[::-1])[::-1]
    p = np.where(prev >= 0, prev, nxt)
    n = np.where(nxt < t, nxt, prev)
    span = np.maximum(n - p, 1)
    w = np.where(n != p, (idx - p) / span, 0.0)[:, None, None]
    return a[p] * (1 - w) + a[n] * w


def assign_by_wrist(hands, pose_wrists):
    """hands: T x 2 x 21 x 3 (NaN when absent) -> same, with each hand moved to the slot of its nearer pose wrist."""
    out = np.full_like(hands, np.nan)
    for f in range(len(hands)):
        found = [(s, hands[f, s]) for s in (0, 1) if np.isfinite(hands[f, s, 0, 0])]
        w = pose_wrists[f]
        if not found:
            continue
        if not np.isfinite(w).all():
            for s, h in found:
                out[f, s] = h
            continue
        cost = np.array([[np.linalg.norm(h[0, :2] - w[k, :2]) for k in (0, 1)] for _, h in found])
        if len(found) == 1:
            out[f, int(cost[0].argmin())] = found[0][1]
        elif cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0]:
            out[f, 0], out[f, 1] = found[0][1], found[1][1]
        else:
            out[f, 1], out[f, 0] = found[0][1], found[1][1]
    return out


def normalise(pose33, hands, face, world, wh):
    """Raw arrays (NaN = unseen) -> points T x P x 3, world T x 2 x 21 x 3, seen T x 4, all float32."""
    t = len(pose33)
    w, h = float(wh[0]), float(wh[1])
    side = max(w, h)
    aspect = np.array([w / side, h / side, w / side])
    pts = np.full((t, P, 3), np.nan)
    pts[:, :NPOSE] = pose33[:, POSE_KEEP, :3]
    pts[:, LH[0]:LH[1]] = hands[:, 0]
    pts[:, RH[0]:RH[1]] = hands[:, 1]
    if face is not None:
        pts[:, FACE[0]:FACE[1]] = face
    pts *= aspect
    groups = [(0, NPOSE), LH, RH, FACE]
    seen = np.stack([np.isfinite(pts[:, a, 0]) for a, _ in groups], 1)

    a, b = pts[:, SHOULDERS[0]], pts[:, SHOULDERS[1]]
    both = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    if both.any():
        centre = ((a[both] + b[both]) / 2).mean(0)
        scale = float(np.median(np.linalg.norm((a[both] - b[both])[:, :2], axis=1)))
    else:
        centre = np.nanmedian(pts.reshape(-1, 3), axis=0) if np.isfinite(pts).any() else np.zeros(3)
        scale = 0.25
    pts = (pts - centre) / max(scale, 1e-2)
    for g, (lo, hi) in enumerate(groups):
        pts[:, lo:hi] = fill_group(pts[:, lo:hi], seen[:, g])
    pts = np.nan_to_num(np.clip(pts, -CLAMP, CLAMP))

    # Holistic reports hand world points in the body's frame (wrist ~0.3 m from the origin): make them wrist-relative.
    wld = np.zeros((t, 2, 21, 3))
    if world is not None:
        for s in (0, 1):
            ok = np.isfinite(world[:, s, 0, 0])
            rel = world[:, s] - world[:, s, :1]
            wld[:, s] = fill_group(np.nan_to_num(rel) * WORLD_SCALE, ok)
        wld = np.clip(wld, -CLAMP, CLAMP)
    return pts.astype(np.float32), wld.astype(np.float32), seen.astype(np.float32)


def load_clip(path, fmt):
    d = np.load(path)
    if fmt == "holistic":
        pose = d["pose"].astype(np.float64)
        return normalise(pose, d["hands"].astype(np.float64), d["face"].astype(np.float64),
                         d["hands_world"].astype(np.float64), d["wh"])
    kp = d["kp"].astype(np.float64)  # legacy: 25 pose + slot hands labelled by MediaPipe handedness
    pose = np.full((len(kp), 33, 3), np.nan)
    pose[:, :25] = kp[:, :25]
    hands = np.stack([kp[:, 25:46], kp[:, 46:67]], 1)
    hands = assign_by_wrist(hands, pose[:, [15, 16], :2])
    return normalise(pose, hands, None, None, d["wh"])


def face_flip_perm(points, seen):
    """Mirror permutation of the 62 face points, found from the mean face: each point's partner is the
    point nearest its reflection about the face's vertical midline."""
    m = seen[..., 3] > 0
    if m.sum() < 100:
        return np.arange(FACE[1] - FACE[0])
    f = points[m][:, FACE[0]:FACE[1], :2].mean(0)
    mid = f[:, 0].mean()
    mirrored = f.copy()
    mirrored[:, 0] = 2 * mid - f[:, 0]
    d = np.linalg.norm(f[None, :, :] - mirrored[:, None, :], axis=-1)
    return d.argmin(1)


def build(kp_dir, fmt, keep=None, t_max=T_MAX):
    import torch
    files = sorted(p for p in pathlib.Path(kp_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if keep is not None:
        files = [p for p in files if p.stem.split("__a")[0] in keep]
    n = len(files)
    X = np.zeros((n, t_max, P, 3), np.float16)
    W = np.zeros((n, t_max, 2, 21, 3), np.float16)
    S = np.zeros((n, t_max, 4), np.float16)
    L = np.zeros(n, np.int64)
    vids = []
    for k, f in enumerate(files):
        pts, wld, seen = load_clip(f, fmt)
        if len(pts) > t_max:  # thin evenly rather than cut the end
            sel = np.linspace(0, len(pts) - 1, t_max).round().astype(int)
            pts, wld, seen = pts[sel], wld[sel], seen[sel]
        t = len(pts)
        X[k, :t], W[k, :t], S[k, :t], L[k] = pts, wld, seen, t
        vids.append(f.stem)
        if (k + 1) % 2000 == 0:
            print(f"{k + 1}/{n}", flush=True)
    sample = slice(0, min(n, 3000))
    perm = face_flip_perm(X[sample].astype(np.float32).reshape(-1, P, 3), S[sample].astype(np.float32).reshape(-1, 4))
    return dict(X=torch.from_numpy(X), W=torch.from_numpy(W), S=torch.from_numpy(S), L=torch.from_numpy(L),
                vid=vids, format=fmt, face_flip=torch.from_numpy(np.asarray(perm, np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kp_dir")
    ap.add_argument("out")
    ap.add_argument("--format", choices=["holistic", "legacy"], default="holistic")
    ap.add_argument("--index", help="with --top: keep only clips of the top-K glosses")
    ap.add_argument("--top", type=int)
    a = ap.parse_args()
    import json
    import torch
    keep = None
    if a.index and a.top:
        keep = {i["video_id"] for e in json.load(open(a.index, encoding="utf-8"))[:a.top] for i in e["instances"]}
    c = build(a.kp_dir, a.format, keep)
    torch.save(c, a.out)
    print(f"{len(c['vid'])} clips -> {a.out}; mean length {c['L'].float().mean():.1f}; "
          f"seen pose/left/right/face {[round(float(v), 3) for v in (c['S'].float().sum((0, 1)) / c['L'].sum())]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
