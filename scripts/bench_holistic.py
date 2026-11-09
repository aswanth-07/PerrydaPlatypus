"""Compare hand detection and cost of three extractors on the same test clips.

current:  pose lite + whole-frame hand landmarker (VIDEO mode), what the pipeline uses now
holistic: MediaPipe HolisticLandmarker, which places hand boxes from the pose (VIDEO mode)
crops:    pose lite, then the hand landmarker in IMAGE mode on an upscaled box around each pose wrist

Usage: bench_holistic.py <index.json> <videos_dir> <models_dir> [--clips 30] [--classes 100] [--pin]
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import mediapipe as mp
import numpy as np
import psutil
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import keypoints as kp

CROP = 224


def make_current(models):
    return kp.make_detectors(models)


def make_holistic(models):
    buf = (models / "holistic_landmarker.task").read_bytes()
    return vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3, min_pose_landmarks_confidence=0.3, min_hand_landmarks_confidence=0.3,
        min_face_detection_confidence=0.3, min_face_landmarks_confidence=0.3))


def make_crops(models):
    hand_buf = (models / "hand_landmarker.task").read_bytes()
    pose_buf = (models / "pose_landmarker_lite.task").read_bytes()
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=hand_buf), running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=pose_buf), running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))
    return hands, pose


def hand_box(pose_lms, side, w, h):
    """Square pixel box around one hand from the pose wrist, pinky and index points (side 0 = person's right)."""
    wrist, pinky, index = (16, 18, 20) if side == 0 else (15, 17, 19)
    p = np.array([[pose_lms[i].x * w, pose_lms[i].y * h] for i in (wrist, pinky, index)])
    centre = p[0] + 0.9 * (p[1:].mean(0) - p[0])
    shoulder = np.hypot((pose_lms[11].x - pose_lms[12].x) * w, (pose_lms[11].y - pose_lms[12].y) * h)
    size = max(0.75 * shoulder, 4.0 * np.linalg.norm(p[1:].mean(0) - p[0]), 32.0)
    return centre, size


def crop_square(rgb, centre, size):
    x0, y0 = centre - size / 2
    m = np.array([[CROP / size, 0, -x0 * CROP / size], [0, CROP / size, -y0 * CROP / size]], dtype=np.float32)
    return cv2.warpAffine(rgb, m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), x0, y0


def run(name, clips, models):
    hand_ms = []
    frames_total = any_hand = both = 0
    for frames, fps in clips:
        step = 1000.0 / max(fps, 1.0)
        if name == "current":
            hands, pose = make_current(models)
        elif name == "holistic":
            hol = make_holistic(models)
        else:
            hands, pose = make_crops(models)
        for i, rgb in enumerate(frames):
            ts = int(i * step)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t = time.perf_counter()
            if name == "current":
                kp_ = kp.frame_keypoints(hands, pose, rgb, ts)
                seen = [not np.isnan(kp_[kp.NUM_POSE, 0]), not np.isnan(kp_[kp.NUM_POSE + kp.NUM_HAND, 0])]
            elif name == "holistic":
                r = hol.detect_for_video(image, ts)
                seen = [bool(r.left_hand_landmarks), bool(r.right_hand_landmarks)]
            else:
                pr = pose.detect_for_video(image, ts)
                seen = [False, False]
                if pr.pose_landmarks:
                    h, w = rgb.shape[:2]
                    for side in (0, 1):
                        centre, size = hand_box(pr.pose_landmarks[0], side, w, h)
                        patch, _, _ = crop_square(rgb, centre, size)
                        hr = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(patch)))
                        seen[side] = bool(hr.hand_landmarks)
            hand_ms.append(1000 * (time.perf_counter() - t))
            frames_total += 1
            any_hand += any(seen)
            both += all(seen)
        if name == "holistic":
            hol.close()
        else:
            hands.close()
            pose.close()
    v = sorted(hand_ms)
    return dict(extractor=name, frames=frames_total, ms_per_frame_p50=round(v[len(v) // 2], 2),
                ms_per_frame_mean=round(sum(v) / len(v), 2), ms_per_frame_p95=round(v[int(0.95 * len(v))], 2),
                any_hand_pct=round(100 * any_hand / frames_total, 1), both_hands_pct=round(100 * both / frames_total, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("videos")
    ap.add_argument("models")
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--only", default="current,holistic,crops")
    a = ap.parse_args()
    if a.pin:
        psutil.Process().cpu_affinity([0])
    entries = json.load(open(a.index, encoding="utf-8"))[:a.classes]
    vdir = pathlib.Path(a.videos)
    paths = [vdir / f"{i['video_id']}.mp4" for e in entries for i in e["instances"]
             if i["split"] == "test" and (vdir / f"{i['video_id']}.mp4").exists()]
    np.random.default_rng(7).shuffle(paths)
    clips = [kp.read_frames(p, limit=64) for p in paths[:a.clips]]
    clips = [c for c in clips if c[0]]
    models = pathlib.Path(a.models)
    print(json.dumps(dict(videos=str(vdir), clips=len(clips), pinned=a.pin, frame_shape=clips[0][0][0].shape)), flush=True)
    for name in a.only.split(","):
        print(json.dumps(run(name, clips, models)), flush=True)


if __name__ == "__main__":
    main()
