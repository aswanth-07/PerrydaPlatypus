"""Holistic keypoints for downloaded ASL Citizen clips, in vocabulary order, skipping clips already done.

Run it repeatedly while aslc_fetch.py is still downloading: each pass takes whatever has arrived for
classes below --max-cls. Output files are aslc_<video stem>.npz so they never collide with WLASL ids.

Usage: extract_aslc.py <aslc_dir> <out_dir> <models_dir> [--max-cls 100] [--limit 32] [--workers 18]
"""
import argparse
import json
import multiprocessing as mp_proc
import os
import pathlib
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import extract_v2 as ev  # noqa: E402


_worker = {"hol": None, "size": None, "t0": 0}


def work(job):
    """Creating the landmarker costs about 0.55 s, as much as landmarking a short clip, so each worker keeps
    one while the frame size stays the same (MediaPipe's smoothing needs a fixed size) and carries the
    VIDEO-mode timestamps forward with a large gap between clips."""
    import numpy as np
    import holistic_kp as hk
    src, dst, limit, models = job
    if dst.exists():
        return "skip", 0
    try:
        frames, fps = ev.read_frames(src, None, limit)
        if not frames:
            return "undecodable", 0
        h, w = frames[0].shape[:2]
        if _worker["hol"] is None or _worker["size"] != (w, h):
            if _worker["hol"] is not None:
                _worker["hol"].close()
            _worker.update(hol=hk.make_holistic(models), size=(w, h), t0=0)
        step = 1000.0 / max(fps, 1.0)
        out = [hk.frame_keypoints(_worker["hol"], f, _worker["t0"] + int(i * step)) for i, f in enumerate(frames)]
        _worker["t0"] += int(len(frames) * step) + 60_000
        arrays = hk.stack_frames(out)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in arrays.items()}, fps=fps, wh=np.array([w, h]))
        os.replace(tmp, dst)
        return "ok", len(frames)
    except Exception as e:
        return f"error: {e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aslc")
    ap.add_argument("out")
    ap.add_argument("models")
    ap.add_argument("--max-cls", type=int, default=100)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    root = pathlib.Path(a.aslc)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [c for c in json.load(open(root / "manifest.json", encoding="utf-8")) if c["cls"] < a.max_cls]
    jobs = [(root / "videos" / c["file"], out / f"aslc_{pathlib.Path(c['file']).stem}.npz", a.limit, a.models)
            for c in clips if (root / "videos" / c["file"]).exists()]
    print(f"{len(clips)} clips below class {a.max_cls}, {len(jobs)} downloaded", flush=True)
    t0, counts, frames = time.time(), {}, 0
    ev.init(a.models, a.out, a.limit)
    with mp_proc.Pool(a.workers, initializer=ev.init, initargs=(a.models, a.out, a.limit)) as pool:
        for k, (status, n) in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            head = status.split(":")[0]
            counts[head] = counts.get(head, 0) + 1
            frames += n
            if k % 1000 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {counts} frames={frames} {time.time() - t0:.0f}s", flush=True)
    print("done", counts, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
