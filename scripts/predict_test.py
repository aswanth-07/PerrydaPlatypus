"""Run the shipped predict() (src/main.py) end to end on the raw WLASL test videos: decode, Holistic,
normalise, INT8 classifier, exactly as the scorer will call it. Writes predictions.jsonl and a summary
with top-1/top-5 and whole-clip latency.

WLASL test entries that point into a longer video are cut to their frame range first (written once to
--cut-dir), because the scored clips each hold one sign.

Usage: predict_test.py <index.json> <videos> <main_videos> <model_dir> <out.jsonl> [--subset 100] [--clips 0]
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import extract_v2 as ev  # noqa: E402

spec = importlib.util.spec_from_file_location("echo_main", ROOT / "src" / "main.py")
echo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(echo)


def cut(src, keep, dst):
    import cv2
    if dst.exists():
        return dst
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or i >= keep[1]:
            break
        if i >= keep[0]:
            out.write(bgr)
        i += 1
    cap.release()
    out.release()
    tmp.replace(dst)
    return dst


def pct(values, q):
    values = sorted(values)
    return round(values[min(int(q * len(values)), len(values) - 1)], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("main_videos")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--clips", type=int, default=0, help="score only the first N test clips (0 = all)")
    ap.add_argument("--cut-dir", default="data/wlasl_test_cut")
    ap.add_argument("--fresh", action="store_true", help="build a new Holistic landmarker for every clip")
    a = ap.parse_args()

    entries = json.load(open(a.index, encoding="utf-8"))[:a.subset]
    vdir, mdir, cdir = pathlib.Path(a.videos), pathlib.Path(a.main_videos), pathlib.Path(a.cut_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for e in entries:
        for i in e["instances"]:
            p = vdir / f"{i['video_id']}.mp4"
            if i["split"] == "test" and p.exists():
                jobs.append((dict(video=str(p), vid=i["video_id"], start=i["frame_start"], end=i["frame_end"],
                                  main=str(mdir / f"{i['video_id']}.mp4")), e["gloss"].upper()))
    jobs = jobs[:a.clips] if a.clips else jobs

    pred = echo.Predictor(a.model_dir, reuse_landmarker=not a.fresh)
    rows, hit1, hit5, cuts = [], 0, 0, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for job, truth in jobs:
            keep = ev.slice_for(job)
            path = pathlib.Path(job["video"])
            if keep:
                path, cuts = cut(path, keep, cdir / path.name), cuts + 1
            r = pred(path)
            f.write(json.dumps(r) + "\n")
            hit1 += r["prediction"] == truth
            hit5 += truth in r["top5"]
            rows.append(dict(r, **pred.last_stages))
    lat = [r["latency_ms"] for r in rows]
    summary = dict(model_dir=a.model_dir, landmarker="fresh per clip" if a.fresh else "reused", clips=len(rows), cut_from_longer_videos=cuts,
                   top1=round(100 * hit1 / len(rows), 2), top5=round(100 * hit5 / len(rows), 2),
                   whole_clip_ms=dict(p50=pct(lat, 0.5), p95=pct(lat, 0.95)),
                   decode_ms_p50=pct([r["decode_ms"] for r in rows], 0.5),
                   keypoints_ms=dict(p50=pct([r["keypoints_ms"] for r in rows], 0.5), p95=pct([r["keypoints_ms"] for r in rows], 0.95)),
                   keypoints_per_frame_ms_p50=pct([r["keypoints_ms"] / max(r["frames"], 1) for r in rows], 0.5),
                   classifier_ms=dict(p50=pct([r["classifier_ms"] for r in rows], 0.5), p95=pct([r["classifier_ms"] for r in rows], 0.95)),
                   frames_p50=pct([r["frames"] for r in rows], 0.5))
    pathlib.Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
