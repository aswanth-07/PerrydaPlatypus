"""Save the middle frame of the same clips from both WLASL copies, side by side.

Usage: compare_copies.py <index.json> <main_videos> <other_videos> <out.png>
"""
import json
import pathlib
import random
import sys

import cv2
import numpy as np


def middle(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    cap.release()
    return frame


def main():
    index, main_dir, other_dir, out = sys.argv[1:5]
    main_dir, other_dir = pathlib.Path(main_dir), pathlib.Path(other_dir)
    entries = json.load(open(index, encoding="utf-8"))
    meta = {i["video_id"]: (e["gloss"], i["source"], i["bbox"]) for e in entries for i in e["instances"]}
    same = sorted({p.stem for p in main_dir.glob("*.mp4")} & {p.stem for p in other_dir.glob("*.mp4")})
    random.Random(11).shuffle(same)
    height = 240
    rows = []
    for vid in same[:5]:
        a, b = middle(main_dir / f"{vid}.mp4"), middle(other_dir / f"{vid}.mp4")
        print(vid, meta[vid], "main", a.shape[:2], "other", b.shape[:2])
        ra = cv2.resize(a, (int(a.shape[1] * height / a.shape[0]), height))
        rb = cv2.resize(b, (int(b.shape[1] * height / b.shape[0]), height))
        row = np.zeros((height, 760, 3), np.uint8)
        row[:, :min(ra.shape[1], 480)] = ra[:, :480]
        row[:, 500:500 + min(rb.shape[1], 260)] = rb[:, :260]
        rows.append(row)
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    main()
