"""Profile WLASL against the official index: what is on disk, per subset and split, and whether it decodes.

Usage: profile_wlasl.py <index.json> <videos_dir> [<videos_dir> ...] [--sample N] [--out out.json]
"""
import argparse
import collections
import json
import pathlib
import random
import statistics as st

import cv2

DECK_EXAMPLES = ["happy", "hand", "house", "help", "home"]
SUBSETS = [100, 300, 1000, 2000]


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return dict(fps=fps, w=w, h=h, frames=n) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--out")
    a = ap.parse_args()

    entries = json.load(open(a.index, encoding="utf-8"))
    report = dict(glosses=len(entries), instances=sum(len(e["instances"]) for e in entries))
    report["signers"] = len({i["signer_id"] for e in entries for i in e["instances"]})
    report["sources"] = collections.Counter(i["source"] for e in entries for i in e["instances"]).most_common(8)

    names = {e["gloss"].lower(): k for k, e in enumerate(entries)}
    report["deck_examples"] = {w: (names[w] if w in names else None) for w in DECK_EXAMPLES}
    report["copies"] = {}

    for vdir in a.videos:
        vdir = pathlib.Path(vdir)
        on_disk = {p.stem: p for p in vdir.glob("*.mp4")}
        indexed = {i["video_id"] for e in entries for i in e["instances"]}
        copy = dict(files=len(on_disk), in_index=len(on_disk.keys() & indexed),
                    not_in_index=len(on_disk.keys() - indexed),
                    bytes=sum(p.stat().st_size for p in on_disk.values()))
        for k in SUBSETS:
            per_split = collections.Counter()
            have = collections.Counter()
            per_class = []
            for e in entries[:k]:
                c = 0
                for i in e["instances"]:
                    per_split[i["split"]] += 1
                    if i["video_id"] in on_disk:
                        have[i["split"]] += 1
                        if i["split"] == "train":
                            c += 1
                per_class.append(c)
            copy[f"wlasl{k}"] = dict(
                indexed=dict(per_split), on_disk=dict(have),
                train_per_class_median=st.median(per_class),
                train_per_class_min=min(per_class),
                classes_without_train=sum(c == 0 for c in per_class))
        rng = random.Random(7)
        sample = rng.sample(sorted(on_disk.values()), min(a.sample, len(on_disk)))
        probes = [probe(p) for p in sample]
        good = [p for p in probes if p]
        copy["decode"] = dict(sampled=len(sample), failed=len(sample) - len(good))
        if good:
            copy["decode"].update(
                fps_median=st.median(p["fps"] for p in good),
                frames_median=st.median(p["frames"] for p in good),
                frames_p95=sorted(p["frames"] for p in good)[int(0.95 * len(good)) - 1],
                seconds_median=round(st.median(p["frames"] / max(p["fps"], 1) for p in good), 2),
                height_median=st.median(p["h"] for p in good),
                width_median=st.median(p["w"] for p in good))
        report["copies"][str(vdir)] = copy

    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        pathlib.Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
