"""Demonstrations for the app's Learn mode: one keypoint clip per word, drawn on the phone as a moving stick figure.

The app shows how a sign is made, the learner copies it in front of the camera, and the live recogniser checks
it. No video is shipped: for each word one training clip's keypoints are kept (nose, shoulders, elbows, wrists,
hips and both hands, 51 points in 2D), cut to the sign itself with a short lead-in and lead-out, resampled to
--frames steps and stored as bytes in a body-centred frame.

The clip chosen for a word is, among its WLASL train and val clips of the word's most common variant, the one
the shipped classifier is most sure of, with a hand in view for most of the sign. Test clips are never used.

File (android/models/<vocab>/demos.bin, packed into the app's assets):
  "EGD1", uint16 count, uint8 frames T, uint8 points P, then per word:
  uint8 name length, the name (UTF-8, upper case), uint16 duration in ms, T*P*2 bytes (x, y in 0..254 of a unit
  square with the shoulders' midpoint at (0.5, 0.45) and the shoulder width 0.18; 255, 255 = point not shown).
Points: nose, left/right shoulder, elbow, wrist, hip (9), left hand 21, right hand 21.

Usage: make_demos.py <index.json> <kp_dir> <cache.pt> <ckpt.pt> <out.bin> [--subset 2000] [--words 300] [--frames 24]
"""
import argparse
import collections
import json
import pathlib
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import contsign as cs
import model as mv
import train as tv

POSE = (0, 11, 12, 13, 14, 15, 16, 23, 24)
CENTRE, WIDTH = np.array([0.5, 0.45]), 0.18


def fill(a):
    """a: T x K x 2 with NaN rows where the group was not seen: linear interpolation over time, ends held."""
    seen = np.isfinite(a[:, 0, 0])
    if not seen.any():
        return a
    t = np.arange(len(a))
    out = a.copy()
    for k in range(a.shape[1]):
        for c in range(2):
            out[:, k, c] = np.interp(t, t[seen], a[seen, k, c])
    return out


def demo(path, frames, lead=4, min_core=10, min_hand=0.7, min_shown=0.25):
    d = np.load(path)
    raw = {k: d[k].astype(np.float64) for k in ("pose", "hands", "face", "hands_world")}
    a = cs.anchor(raw, d["wh"])
    if a is None:
        return None
    s0, lo, hi, s1 = cs.core_span(cs.heights(a))
    if hi - lo < min_core:
        return None
    hand = np.isfinite(a["hands"][lo:hi, :, 0, 0])
    if hand.any(1).mean() < min_hand:
        return None
    lo, hi = max(lo - lead, 0), min(hi + lead, len(a["pose"]))
    pose = fill(a["pose"][lo:hi][:, POSE, :2])
    parts = [pose]
    for slot in (0, 1):
        h = a["hands"][lo:hi, slot, :, :2]
        shown = np.isfinite(h[:, 0, 0]).mean()
        parts.append(fill(h) if shown >= min_shown else np.full_like(h, np.nan))
    pts = np.concatenate(parts, 1)                                  # T x 51 x 2, anchored unit square
    pts = (pts - cs.ANCHOR_C[:2]) / cs.ANCHOR_W * WIDTH + CENTRE
    pos = np.linspace(0, len(pts) - 1, frames)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(pts) - 1)
    w = (pos - i0)[:, None, None]
    pts = pts[i0] * (1 - w) + pts[i1] * w
    q = np.where(np.isfinite(pts), np.clip(np.round(pts * 254), 0, 254), 255).astype(np.uint8)
    q[(q == 255).any(-1)] = 255
    return q, int(round(1000 * (hi - lo) / float(d["fps"]))), float(hand.any(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("cache")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--words", type=int, default=300, help="the first this many words of the index (WLASL orders them by clip count)")
    ap.add_argument("--frames", type=int, default=24)
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))
    kp = pathlib.Path(a.kp_dir)

    # How sure the shipped classifier is of each train/val clip's own word.
    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, a.subset)
    ids = torch.cat([rows["train"][0], rows["val"][0]])
    y = torch.cat([rows["train"][1], rows["val"][1]])
    keep = (y < a.words) & torch.tensor(["__a" not in cache["vid"][int(r)] for r in ids])
    ids, y = ids[keep], y[keep]
    ck = torch.load(a.ckpt, weights_only=False)
    model = mv.build(ck["cfg"], ck["classes"]).cuda().eval()
    model.load_state_dict(ck["state"])
    data = {k: cache[k].cuda() for k in ("X", "W", "S", "L")}
    perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).cuda()
    with torch.no_grad():
        p = tv.predict_logits(model, data, ids.cuda(), ck["cfg"], False, perm).cpu()
    sure = {cache["vid"][int(r)]: float(p[n, int(y[n])]) for n, r in enumerate(ids)}

    out, skipped = [], []
    for e in entries[:a.words]:
        inst = [i for i in e["instances"] if i["split"] in ("train", "val") and i["video_id"] in sure and (kp / f"{i['video_id']}.npz").exists()]
        common = collections.Counter(i.get("variation_id", 0) for i in inst).most_common(1)
        best = None
        for relaxed in (False, True):       # the most common variant first; any variant if none of its clips will do
            for i in sorted(inst, key=lambda i: -sure[i["video_id"]]):
                if not relaxed and common and i.get("variation_id", 0) != common[0][0]:
                    continue
                d = demo(kp / f"{i['video_id']}.npz", a.frames, min_hand=0.5 if relaxed else 0.7)
                if d is not None:
                    best = (i["video_id"], d)
                    break
            if best:
                break
        if best is None:
            skipped.append(e["gloss"])
            continue
        out.append((e["gloss"].upper(), best[1][1], best[1][0], best[0], sure[best[0]]))

    blob = bytearray(b"EGD1" + struct.pack("<HBB", len(out), a.frames, out[0][2].shape[1]))
    for name, ms, q, _, _ in out:
        nb = name.encode("utf-8")
        blob += struct.pack("<B", len(nb)) + nb + struct.pack("<H", min(ms, 65535)) + q.tobytes()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_bytes(bytes(blob))
    info = dict(words=len(out), skipped=skipped, frames=a.frames, bytes=len(blob),
                median_ms=int(np.median([o[1] for o in out])), mean_sure=round(float(np.mean([o[4] for o in out])), 3),
                clips={o[0]: o[3] for o in out})
    pathlib.Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print({k: v for k, v in info.items() if k != "clips"})


if __name__ == "__main__":
    main()
