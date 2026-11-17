"""Extract Holistic keypoints for WLASL clips, honouring the index's frame_start/frame_end. Safe to re-run.

Some WLASL entries are cut from longer videos. The 21k copy is already cut for most of them, but 74 clips
still hold the whole source video, so the next sign leaks in; those are sliced here, with frame numbers
scaled by the copy's frame rate against the main copy's (or 25/30 when the main copy lacks the clip).

Usage: extract.py <index.json> <videos_dir> <out_dir> <models_dir> [--main-videos DIR] [--workers N] [--top K] [--limit 96]
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


def init(models_dir, out_dir, limit):
    import cv2
    import psutil
    cv2.setNumThreads(1)
    if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    _state.update(models=models_dir, out=pathlib.Path(out_dir), limit=limit)


def frame_count(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def slice_for(job):
    """Frame range [a, b) to keep, or None for the whole clip."""
    path, start, end, main_path = job["video"], job["start"], job["end"], job["main"]
    n = frame_count(path)
    if start == 1 and end == -1:
        return None
    length = end - start + 1 if end != -1 else None
    if length is not None and abs(n - length) <= 2:
        return None  # already cut
    if (end != -1 and n >= end) or (end == -1 and n > start):
        ratio = 25.0 / 30.0
        if main_path and pathlib.Path(main_path).exists():
            m = frame_count(main_path)
            if m > 0:
                ratio = n / m
        a = max(int(round((start - 1) * ratio)), 0)
        b = n if end == -1 else min(int(round(end * ratio)), n)
        return (a, b) if b - a >= 8 else None
    return None  # already cut at another frame rate


def read_frames(path, keep, limit, max_side=480):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if keep is None or keep[0] <= i < keep[1]:
            scale = max_side / max(bgr.shape[:2])
            if scale < 1:
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
        if keep is not None and i >= keep[1]:
            break
    cap.release()
    if len(frames) > limit:
        idx = np.linspace(0, len(frames) - 1, limit).round().astype(int)
        frames = [frames[j] for j in idx]
    return frames, fps


def work(job):
    import numpy as np
    import holistic_kp as hk
    dst = _state["out"] / f"{job['vid']}.npz"
    if dst.exists():
        return job["vid"], "skip", 0
    try:
        keep = slice_for(job)
        frames, fps = read_frames(job["video"], keep, _state["limit"])
        if not frames:
            return job["vid"], "undecodable", 0
        with hk.make_holistic(_state["models"]) as hol:
            arrays = hk.clip_keypoints(hol, frames, fps)
        h, w = frames[0].shape[:2]
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in arrays.items()}, fps=fps, wh=np.array([w, h]),
                            keep=np.array(keep if keep else (-1, -1)))
        os.replace(tmp, dst)
        return job["vid"], "sliced" if keep else "ok", len(frames)
    except Exception as e:  # keep the sweep going; bad clips are reported at the end
        return job["vid"], f"error: {e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("out")
    ap.add_argument("models")
    ap.add_argument("--main-videos", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=96)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = json.load(open(a.index, encoding="utf-8"))[:a.top]
    vdir = pathlib.Path(a.videos)
    mdir = pathlib.Path(a.main_videos) if a.main_videos else None
    jobs = []
    for e in entries:
        for i in e["instances"]:
            p = vdir / f"{i['video_id']}.mp4"
            if p.exists():
                jobs.append(dict(video=str(p), vid=i["video_id"], start=i["frame_start"], end=i["frame_end"],
                                 main=str(mdir / f"{i['video_id']}.mp4") if mdir else None))
    print(f"{len(jobs)} clips, {a.workers} workers", flush=True)
    t0 = time.time()
    counts, frames, bad = {}, 0, []
    with mp_proc.Pool(a.workers, initializer=init, initargs=(a.models, a.out, a.limit)) as pool:
        for k, (vid, status, n) in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            head = status.split(":")[0]
            counts[head] = counts.get(head, 0) + 1
            frames += n
            if head not in ("ok", "skip", "sliced"):
                bad.append((vid, status))
            if k % 500 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {counts} frames={frames} {time.time() - t0:.0f}s", flush=True)
    (out / "_bad.json").write_text(json.dumps(bad, indent=0), encoding="utf-8")
    print("done", counts, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
