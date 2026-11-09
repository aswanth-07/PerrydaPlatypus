"""MediaPipe keypoint extraction shared by batch extraction and the predict() prototype.

One frame becomes 67 landmarks x 3: 25 upper-body pose points plus 21 points per hand.
Missing parts stay NaN so the model can tell "not seen" from "at the origin".
"""
import pathlib

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

NUM_POSE = 25
NUM_HAND = 21
NUM_POINTS = NUM_POSE + 2 * NUM_HAND
MAX_FRAMES = 64


def make_detectors(models_dir, threads_hint=None):
    models_dir = pathlib.Path(models_dir)
    # Read the files ourselves: MediaPipe's own path handling is fragile on long Windows paths.
    hand_buf = (models_dir / "hand_landmarker.task").read_bytes()
    pose_buf = (models_dir / "pose_landmarker_lite.task").read_bytes()
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=hand_buf),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=pose_buf),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))
    return hands, pose


def frame_keypoints(hands, pose, rgb, ts_ms):
    out = np.full((NUM_POINTS, 3), np.nan, dtype=np.float32)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pr = pose.detect_for_video(image, ts_ms)
    if pr.pose_landmarks:
        for k, lm in enumerate(pr.pose_landmarks[0][:NUM_POSE]):
            out[k] = (lm.x, lm.y, lm.z)
    hr = hands.detect_for_video(image, ts_ms)
    for lms, handed in zip(hr.hand_landmarks, hr.handedness):
        # MediaPipe assumes mirrored input, so its "Left" is the signer's right hand here.
        base = NUM_POSE if handed[0].category_name == "Left" else NUM_POSE + NUM_HAND
        for k, lm in enumerate(lms):
            out[base + k] = (lm.x, lm.y, lm.z)
    return out


def sample_indices(total, limit=MAX_FRAMES):
    if total <= limit:
        return np.arange(total)
    return np.linspace(0, total - 1, limit).round().astype(int)


def read_frames(video_path, limit=MAX_FRAMES, max_side=480):
    """Decode a clip into at most `limit` evenly spaced RGB frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        scale = max_side / max(bgr.shape[:2])
        if scale < 1:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return [], fps
    return [frames[i] for i in sample_indices(len(frames), limit)], fps


def clip_keypoints(hands, pose, frames, fps, start_ms=0):
    step = 1000.0 / max(fps, 1.0)
    return np.stack([frame_keypoints(hands, pose, f, int(start_ms + i * step)) for i, f in enumerate(frames)])
