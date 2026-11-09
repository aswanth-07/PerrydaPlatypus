"""Split missing hands into true misses (pose wrist inside the frame) and hands out of view.

Usage: hand_misses.py <index.json> <kp_dir> [--classes 100] [--split test]
"""
import argparse
import json
import pathlib

import numpy as np

POSE_WRIST = {0: 15, 1: 16}  # hand slot 0 sits on pose 15, the person's left wrist (measured, 89% of frames)
SLOT_BASE = {0: 25, 1: 46}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))[:a.classes]
    kdir = pathlib.Path(a.kp_dir)
    tot = dict(frames=0, mid_frames=0, mid_any=0, wrist_in=0, wrist_in_seen=0, wrist_out=0, wrist_out_seen=0)
    per_slot = {s: dict(wrist_in=0, seen=0) for s in (0, 1)}
    clips = 0
    for e in entries:
        for inst in e["instances"]:
            if inst["split"] != a.split:
                continue
            f = kdir / f"{inst['video_id']}.npz"
            if not f.exists():
                continue
            k = np.load(f)["kp"].astype(np.float32)
            clips += 1
            t = len(k)
            seen = np.stack([~np.isnan(k[:, SLOT_BASE[s], 0]) for s in (0, 1)], 1)
            lo, hi = int(0.25 * t), max(int(0.75 * t), int(0.25 * t) + 1)
            tot["frames"] += t
            tot["mid_frames"] += hi - lo
            tot["mid_any"] += int(seen[lo:hi].any(1).sum())
            for s in (0, 1):
                w = k[:, POSE_WRIST[s]]
                inside = (~np.isnan(w[:, 0])) & (w[:, 0] > 0.02) & (w[:, 0] < 0.98) & (w[:, 1] > 0.02) & (w[:, 1] < 0.98)
                tot["wrist_in"] += int(inside.sum())
                tot["wrist_in_seen"] += int((inside & seen[:, s]).sum())
                tot["wrist_out"] += int((~inside).sum())
                tot["wrist_out_seen"] += int(((~inside) & seen[:, s]).sum())
                per_slot[s]["wrist_in"] += int(inside.sum())
                per_slot[s]["seen"] += int((inside & seen[:, s]).sum())
    out = dict(clips=clips, frames=tot["frames"],
               middle_half_any_hand_pct=round(100 * tot["mid_any"] / tot["mid_frames"], 1),
               hand_seen_when_wrist_in_frame_pct=round(100 * tot["wrist_in_seen"] / max(tot["wrist_in"], 1), 1),
               hand_seen_when_wrist_out_pct=round(100 * tot["wrist_out_seen"] / max(tot["wrist_out"], 1), 1),
               wrist_in_frame_pct=round(100 * tot["wrist_in"] / (2 * tot["frames"]), 1),
               per_slot_seen_when_in={s: round(100 * v["seen"] / max(v["wrist_in"], 1), 1) for s, v in per_slot.items()})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
