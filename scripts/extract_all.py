"""Extract keypoints for every WLASL clip on disk, most common glosses first. Safe to re-run: done clips are skipped.

Usage: extract_all.py <index.json> <videos_dir> <out_dir> <models_dir> [--workers N] [--top K]
"""
import argparse
import json
import multiprocessing as mp_proc
import os
import pathlib
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_state = {}


def init(models_dir, out_dir):
    import cv2
    import psutil
    cv2.setNumThreads(1)
    if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):  # stay out of the way of whatever else the laptop is doing
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    _state["models"] = models_dir
    _state["out"] = pathlib.Path(out_dir)


def work(job):
    # Detectors are rebuilt per clip: VIDEO mode keeps tracking state and demands rising timestamps.
    import numpy as np
    import keypoints as kp
    video, vid = job
    dst = _state["out"] / f"{vid}.npz"
    if dst.exists():
        return vid, "skip", 0
    try:
        frames, fps = kp.read_frames(video)
        if not frames:
            return vid, "undecodable", 0
        hands, pose = kp.make_detectors(_state["models"])
        with hands, pose:
            pts = kp.clip_keypoints(hands, pose, frames, fps)
        tmp = dst.with_suffix(".tmp.npz")
        h, w = frames[0].shape[:2]
        np.savez_compressed(tmp, kp=pts.astype(np.float16), fps=fps, wh=np.array([w, h]))
        os.replace(tmp, dst)
        return vid, "ok", len(frames)
    except Exception as e:  # keep the sweep going; bad clips are reported at the end
        return vid, f"error: {e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("out")
    ap.add_argument("models")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--top", type=int, default=2000)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = json.load(open(a.index, encoding="utf-8"))[:a.top]
    vdir = pathlib.Path(a.videos)
    jobs = []
    for e in entries:
        for i in e["instances"]:
            p = vdir / f"{i['video_id']}.mp4"
            if p.exists():
                jobs.append((p, i["video_id"]))
    print(f"{len(jobs)} clips, {a.workers} workers", flush=True)
    t0 = time.time()
    counts, frames, bad = {}, 0, []
    with mp_proc.Pool(a.workers, initializer=init, initargs=(a.models, a.out)) as pool:
        for k, (vid, status, n) in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            head = status.split(":")[0]
            counts[head] = counts.get(head, 0) + 1
            frames += n
            if head not in ("ok", "skip"):
                bad.append((vid, status))
            if k % 500 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {counts} frames={frames} {time.time() - t0:.0f}s", flush=True)
    (out / "_bad.json").write_text(json.dumps(bad, indent=0), encoding="utf-8")
    print("done", counts, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
