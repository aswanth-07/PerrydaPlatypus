"""Copy the keypoint files of the top-K glosses into their own folder, so training can run during extraction.

Usage: snapshot_kp.py <index.json> <kp_dir> <out_dir> <top_k>
"""
import json
import pathlib
import shutil
import sys

import numpy as np


def main():
    index, src, dst, top = sys.argv[1:5]
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    entries = json.load(open(index, encoding="utf-8"))[:int(top)]
    copied = missing = 0
    frames = hand_frames = both_frames = pose_frames = 0
    for e in entries:
        for i in e["instances"]:
            p = src / f"{i['video_id']}.npz"
            if not p.exists():
                missing += 1
                continue
            if not (dst / p.name).exists():
                shutil.copy2(p, dst / p.name)
            copied += 1
            kp = np.load(p)["kp"].astype(np.float32)
            left = np.isfinite(kp[:, 25:46, 0]).any(axis=1)
            right = np.isfinite(kp[:, 46:67, 0]).any(axis=1)
            frames += len(kp)
            hand_frames += int((left | right).sum())
            both_frames += int((left & right).sum())
            pose_frames += int(np.isfinite(kp[:, :25, 0]).any(axis=1).sum())
    print(f"clips {copied} missing {missing} frames {frames}")
    print(f"pose seen {100 * pose_frames / max(frames, 1):.1f}%  any hand {100 * hand_frames / max(frames, 1):.1f}%  both hands {100 * both_frames / max(frames, 1):.1f}%")


if __name__ == "__main__":
    main()
