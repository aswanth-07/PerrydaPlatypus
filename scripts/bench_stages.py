"""Time the MediaPipe stages per frame on real clips, in VIDEO mode, and show what cheaper settings cost in detections.

Usage: bench_stages.py <index.json> <videos_dir> <models_dir> [--clips 20] [--classes 100] [--pin]
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "3")

import mediapipe as mp
import numpy as np
import psutil
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import keypoints as kp


def hands(buf, num):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO, num_hands=num,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))


def pose(buf):
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))


def run(variant, clips, hand_buf, pose_buf):
    pose_ms, hand_ms = [], []
    frames_total = seen_any = seen_both = hand_runs = 0
    for frames, fps in clips:
        hd = hands(hand_buf, variant["num_hands"]) if variant["hands_every"] else None
        pd = pose(pose_buf) if variant["pose"] else None
        step = 1000.0 / max(fps, 1.0)
        for i, rgb in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i * step)
            if pd:
                t = time.perf_counter()
                pd.detect_for_video(image, ts)
                pose_ms.append(1000 * (time.perf_counter() - t))
            if hd and i % variant["hands_every"] == 0:
                t = time.perf_counter()
                r = hd.detect_for_video(image, ts)
                hand_ms.append(1000 * (time.perf_counter() - t))
                hand_runs += 1
                seen_any += len(r.hand_landmarks) >= 1
                seen_both += len(r.hand_landmarks) >= 2
            frames_total += 1
        if hd:
            hd.close()
        if pd:
            pd.close()

    def stat(values):
        values = sorted(values)
        return dict(p50=round(values[len(values) // 2], 2), p95=round(values[int(0.95 * len(values))], 2)) if values else None

    per_frame = (sum(pose_ms) + sum(hand_ms)) / max(frames_total, 1)
    return dict(variant=variant["name"], frames=frames_total, pose_ms=stat(pose_ms), hand_ms_per_run=stat(hand_ms),
                mean_keypoint_ms_per_frame=round(per_frame, 2),
                any_hand_pct=round(100 * seen_any / max(hand_runs, 1), 1) if hand_runs else None,
                both_hands_pct=round(100 * seen_both / max(hand_runs, 1), 1) if hand_runs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("models")
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args()
    proc = psutil.Process()
    if a.pin:
        proc.cpu_affinity([0])
    entries = json.load(open(a.index, encoding="utf-8"))[:a.classes]
    vdir = pathlib.Path(a.videos)
    paths = [vdir / f"{i['video_id']}.mp4" for e in entries for i in e["instances"]
             if i["split"] == "test" and (vdir / f"{i['video_id']}.mp4").exists()]
    np.random.default_rng(7).shuffle(paths)
    clips = [kp.read_frames(p, limit=64) for p in paths[:a.clips]]
    models = pathlib.Path(a.models)
    hand_buf = (models / "hand_landmarker.task").read_bytes()
    pose_buf = (models / "pose_landmarker_lite.task").read_bytes()
    variants = [
        dict(name="pose + 2 hands, every frame", pose=True, num_hands=2, hands_every=1),
        dict(name="pose only", pose=True, num_hands=2, hands_every=0),
        dict(name="2 hands only, every frame", pose=False, num_hands=2, hands_every=1),
        dict(name="1 hand only, every frame", pose=False, num_hands=1, hands_every=1),
        dict(name="pose + 2 hands, hands every 2nd frame", pose=True, num_hands=2, hands_every=2),
        dict(name="pose + 2 hands, hands every 3rd frame", pose=True, num_hands=2, hands_every=3),
    ]
    size = clips[0][0][0].shape if clips and clips[0][0] else None
    print(json.dumps(dict(videos=str(vdir), pinned=a.pin, clips=len(clips), frame_shape=size)), flush=True)
    for v in variants:
        print(json.dumps(run(v, clips, hand_buf, pose_buf)), flush=True)
    print(json.dumps(dict(peak_rss_mb=round(proc.memory_info().peak_wset / 1e6, 1))), flush=True)


if __name__ == "__main__":
    main()
