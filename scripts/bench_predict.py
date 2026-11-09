"""Time predict() stage by stage on real test clips, pinned to one core or unpinned.

Usage: bench_predict.py <index.json> <videos_dir> <run_dir> <models_dir> [--clips 60] [--frames 32] [--pin]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import psutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import main as predict_mod


def pct(values, q):
    values = sorted(values)
    return round(values[min(int(q * len(values)), len(values) - 1)], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("run")
    ap.add_argument("models")
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--classes", type=int, default=100)
    a = ap.parse_args()

    proc = psutil.Process()
    if a.pin:
        proc.cpu_affinity([0])
    entries = json.load(open(a.index, encoding="utf-8"))[:a.classes]
    vdir = pathlib.Path(a.videos)
    tests = [(vdir / f"{i['video_id']}.mp4", c) for c, e in enumerate(entries) for i in e["instances"]
             if i["split"] == "test" and (vdir / f"{i['video_id']}.mp4").exists()]
    np.random.default_rng(7).shuffle(tests)
    tests = tests[:a.clips]

    run = pathlib.Path(a.run)
    labels = predict_mod.load_labels(a.index, a.classes)
    pred = predict_mod.Predictor(run / "model_int8.onnx", labels, a.models, frames=a.frames)
    stages, totals, hits = [], [], 0
    for path, c in tests:
        out = pred(path)
        s = dict(pred.last_stages)
        stages.append(s)
        totals.append(out["latency_ms"])
        hits += out["prediction"] == labels[c].upper()

    frame_ms = [s["keypoints_ms"] / max(s["frames"], 1) for s in stages]
    report = dict(
        pinned_to_one_core=a.pin, clips=len(tests), frames_limit=a.frames, top1=round(100 * hits / max(len(tests), 1), 1),
        median_frames=pct([s["frames"] for s in stages], 0.5),
        whole_clip_ms=dict(p50=pct(totals, 0.5), p95=pct(totals, 0.95)),
        keypoints_per_frame_ms=dict(p50=pct(frame_ms, 0.5), p95=pct(frame_ms, 0.95)),
        decode_ms_p50=pct([s["decode_ms"] for s in stages], 0.5),
        load_detectors_ms_p50=pct([s["load_ms"] for s in stages], 0.5),
        features_ms_p50=pct([s["features_ms"] for s in stages], 0.5),
        classify_ms=dict(p50=pct([s["classify_ms"] for s in stages], 0.5), p95=pct([s["classify_ms"] for s in stages], 0.95)),
        peak_rss_mb=round(proc.memory_info().peak_wset / 1e6, 1) if hasattr(proc.memory_info(), "peak_wset") else None,
        sizes_mb={p.name: round(p.stat().st_size / 1e6, 2) for p in [run / "model_int8.onnx", pathlib.Path(a.models) / "hand_landmarker.task", pathlib.Path(a.models) / "pose_landmarker_lite.task"]},
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
