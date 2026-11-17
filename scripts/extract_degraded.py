"""Holistic keypoints from degraded videos (video_aug) for training copies and for the robustness test.

train: each training-split clip of the top-K glosses gets --copies degraded versions, each with its own
       random mix of conditions, saved as <vid>__a<k>.npz.
eval:  each test-split clip gets one version per fixed condition in video_aug.EVAL_SPECS, saved as
       <condition>/<vid>.npz, with seeds that no training copy uses.

Usage: extract_degraded.py train|eval <index.json> <videos_dir> <out_dir> <models_dir> [--top 100] [--copies 2]
       [--main-videos DIR] [--workers N] [--split test]
"""
import argparse
import json
import multiprocessing as mp_proc
import os
import pathlib
import sys
import time
import zlib

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp

import extract as ev  # noqa: E402

_state = {}


def init(models_dir, limit):
    import cv2
    import psutil
    cv2.setNumThreads(1)
    if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
        psutil.Process().nice(getattr(psutil, os.environ.get("ECHO_EXTRACT_PRIORITY", "BELOW_NORMAL_PRIORITY_CLASS")))
    _state.update(models=models_dir, limit=limit)


def work(job):
    import numpy as np
    import holistic_kp as hk
    import video_aug as va
    dst = pathlib.Path(job["dst"])
    if dst.exists():
        return job["dst"], "skip", 0
    try:
        frames, fps = ev.read_frames(job["video"], ev.slice_for(job), _state["limit"])
        if not frames:
            return job["dst"], "undecodable", 0
        frames = va.apply(frames, job["spec"], job["seed"])
        if job.get("enhance"):
            frames = va.enhance(frames)
        with hk.make_holistic(_state["models"]) as hol:
            arrays = hk.clip_keypoints(hol, frames, fps)
        h, w = frames[0].shape[:2]
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in arrays.items()}, fps=fps,
                            wh=np.array([w, h]), spec=json.dumps(job["spec"]))
        os.replace(tmp, dst)
        return job["dst"], "ok", len(frames)
    except Exception as e:
        return job["dst"], f"error: {e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("out")
    ap.add_argument("models")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--main-videos", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=96)
    ap.add_argument("--enhance", action="store_true", help="apply the run-time low-light fix after degrading")
    ap.add_argument("--conditions", default="", help="eval: only these conditions (comma-separated)")
    ap.add_argument("--strength-scale", type=float, default=1.0, help="eval: scale every condition's strength (0.57 turns 0.7 into 0.4)")
    ap.add_argument("--must", default="", help="train: conditions every copy includes (comma-separated), at strength 0.3-1.0")
    ap.add_argument("--copy-offset", type=int, default=0, help="train: first copy index, so new copies get new seeds and names")
    ap.add_argument("--shuffle", action="store_true", help="process jobs in a fixed random order, so a stopped run covers every class")
    a = ap.parse_args()

    import numpy as np
    import video_aug as va
    out = pathlib.Path(a.out)
    entries = json.load(open(a.index, encoding="utf-8"))[:a.top]
    vdir = pathlib.Path(a.videos)
    mdir = pathlib.Path(a.main_videos) if a.main_videos else None
    jobs = []
    for e in entries:
        for i in e["instances"]:
            p = vdir / f"{i['video_id']}.mp4"
            if not p.exists():
                continue
            base = dict(video=str(p), vid=i["video_id"], start=i["frame_start"], end=i["frame_end"],
                        main=str(mdir / f"{i['video_id']}.mp4") if mdir else None)
            h = zlib.crc32(i["video_id"].encode())
            if a.mode == "train" and i["split"] == "train":
                for k in range(a.copy_offset, a.copy_offset + a.copies):
                    seed = h * 31 + k
                    rng = np.random.default_rng(seed)
                    spec = va.random_spec(rng)
                    for c in filter(None, a.must.split(",")):
                        spec.setdefault(c, float(rng.uniform(0.3, 1.0)))
                    jobs.append(dict(base, spec=spec, seed=seed, dst=str(out / f"{i['video_id']}__a{k}.npz")))
            elif a.mode == "eval" and i["split"] == a.split:
                for c, spec in va.EVAL_SPECS.items():
                    if a.conditions and c not in a.conditions.split(","):
                        continue
                    spec = {k: v * a.strength_scale for k, v in spec.items()}
                    (out / c).mkdir(parents=True, exist_ok=True)
                    jobs.append(dict(base, spec=spec, seed=10_000_019 + h, enhance=a.enhance, dst=str(out / c / f"{i['video_id']}.npz")))
    out.mkdir(parents=True, exist_ok=True)
    if a.shuffle:
        jobs = [jobs[i] for i in np.random.default_rng(0).permutation(len(jobs))]
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    counts, frames, bad = {}, 0, []
    with mp_proc.Pool(a.workers, initializer=init, initargs=(a.models, a.limit)) as pool:
        for k, (dst, status, n) in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            head = status.split(":")[0]
            counts[head] = counts.get(head, 0) + 1
            frames += n
            if head not in ("ok", "skip"):
                bad.append((dst, status))
            if k % 500 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {counts} frames={frames} {time.time() - t0:.0f}s", flush=True)
    (out / "_bad.json").write_text(json.dumps(bad, indent=0), encoding="utf-8")
    print("done", counts, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
