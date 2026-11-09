"""Real image-level stress test: damage the test clips' frames, re-run the whole pipeline, measure the accuracy drop.

Conditions match the deck's "moving heads, angled signers, low light" plus occlusion.

Usage: robustness.py <index.json> <videos_dir> <run_dir> <models_dir> [--classes 100] [--workers 12]
"""
import argparse
import json
import multiprocessing as mp_proc
import os
import pathlib
import sys

os.environ.setdefault("GLOG_minloglevel", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def t_clean(f, rng):
    return f


def t_low_light(f, rng):
    import numpy as np
    return (np.clip((f / 255.0) ** 2.0 * 0.45 + rng.normal(0, 0.012, f.shape), 0, 1) * 255).astype(np.uint8)


def t_angled(f, rng):
    import cv2
    import numpy as np
    h, w = f.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.12 * w, 0.08 * h], [w, 0], [w, h], [0.12 * w, 0.92 * h]])
    return cv2.warpPerspective(f, cv2.getPerspectiveTransform(src, dst), (w, h))


def t_roll(f, rng):
    import cv2
    h, w = f.shape[:2]
    return cv2.warpAffine(f, cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0), (w, h))


def t_occlusion(f, rng):
    # A fixed bar across the signer's chest and one side, as a passer-by or an object would.
    g = f.copy()
    h, w = g.shape[:2]
    g[int(0.55 * h):int(0.8 * h), int(0.5 * w):] = 0
    return g


def t_small(f, rng):
    import cv2
    h, w = f.shape[:2]
    s = 180 / max(h, w)
    if s >= 1:
        return f
    return cv2.resize(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)


CONDITIONS = dict(clean=t_clean, low_light=t_low_light, angled=t_angled, roll_15deg=t_roll, occlusion=t_occlusion, low_res=t_small)
_s = {}


def init(run, models, index, classes):
    import cv2
    import main as predict_mod
    cv2.setNumThreads(1)
    labels = predict_mod.load_labels(index, classes)
    _s["labels"] = labels
    _s["pred"] = predict_mod.Predictor(pathlib.Path(run) / "model_int8.onnx", labels, models, frames=64)


def work(job):
    import numpy as np
    path, c, cond = job
    rng = np.random.default_rng(7)
    fn = CONDITIONS[cond]
    try:
        out = _s["pred"](path, transform=lambda f: fn(f, rng))
    except Exception as e:
        return cond, 0, 0, str(e)
    gold = _s["labels"][c].upper()
    return cond, int(out["prediction"] == gold), int(gold in out["top5"]), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("run")
    ap.add_argument("models")
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    entries = json.load(open(a.index, encoding="utf-8"))[:a.classes]
    vdir = pathlib.Path(a.videos)
    tests = [(vdir / f"{i['video_id']}.mp4", c) for c, e in enumerate(entries) for i in e["instances"]
             if i["split"] == "test" and (vdir / f"{i['video_id']}.mp4").exists()]
    jobs = [(p, c, cond) for cond in CONDITIONS for p, c in tests]
    tally = {k: [0, 0, 0] for k in CONDITIONS}
    errors = {}
    with mp_proc.Pool(a.workers, initializer=init, initargs=(a.run, a.models, a.index, a.classes)) as pool:
        for cond, t1, t5, err in pool.imap_unordered(work, jobs, chunksize=4):
            tally[cond][0] += t1
            tally[cond][1] += t5
            tally[cond][2] += 1
            if err:
                errors[err] = errors.get(err, 0) + 1
    clean = 100 * tally["clean"][0] / max(tally["clean"][2], 1)
    report = {k: dict(n=v[2], top1=round(100 * v[0] / max(v[2], 1), 1), top5=round(100 * v[1] / max(v[2], 1), 1),
                      drop_pt=round(clean - 100 * v[0] / max(v[2], 1), 1)) for k, v in tally.items()}
    report["errors"] = errors
    text = json.dumps(report, indent=1)
    (pathlib.Path(a.run) / "robustness.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
