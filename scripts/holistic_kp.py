"""MediaPipe Holistic keypoints: the extractor shared by batch extraction, predict() and the Android port.

Holistic places each hand box from the pose arm instead of searching the whole frame, which finds more
hands in fast signing. One frame is stored as named arrays, all float32 with NaN for "not seen":

  pose        33 x 4   image x, y, z and visibility
  hands        2 x 21 x 3   image x, y, z; slot 0 is the person's left hand, slot 1 the right
  hands_world  2 x 21 x 3   metric 3D hand points from the hand model, centred on the hand
  face        62 x 3   lips, nose, eye and eyebrow points from the face mesh

Hands are put in slots by distance to the pose wrists, not by MediaPipe's handedness label, which
we measured to be wrong on about 10% of frames.
"""
import pathlib

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
NOSE = [1, 2, 98, 327]
EYES = [33, 133, 160, 158, 144, 153, 362, 263, 385, 387, 380, 373, 70, 105, 107, 336, 334, 300]
FACE = LIPS + NOSE + EYES
POSE_WRIST = (15, 16)  # the person's left and right wrist


def make_holistic(models_dir, video=True):
    buf = (pathlib.Path(models_dir) / "holistic_landmarker.task").read_bytes()
    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    return vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=buf), running_mode=mode,
        min_pose_detection_confidence=0.3, min_pose_landmarks_confidence=0.3, min_hand_landmarks_confidence=0.3,
        min_face_detection_confidence=0.3, min_face_landmarks_confidence=0.3))


def empty_frame():
    return dict(pose=np.full((33, 4), np.nan, np.float32), hands=np.full((2, 21, 3), np.nan, np.float32),
                hands_world=np.full((2, 21, 3), np.nan, np.float32), face=np.full((len(FACE), 3), np.nan, np.float32))


def frame_keypoints(holistic, rgb, ts_ms):
    out = empty_frame()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    r = holistic.detect_for_video(image, ts_ms) if ts_ms is not None else holistic.detect(image)
    if r.pose_landmarks:
        out["pose"] = np.array([(p.x, p.y, p.z, p.visibility if p.visibility is not None else 1.0)
                                for p in r.pose_landmarks], np.float32)[:33]
    if r.face_landmarks:
        f = r.face_landmarks
        out["face"] = np.array([(f[i].x, f[i].y, f[i].z) for i in FACE], np.float32)
    found = []
    for lms, world in ((r.left_hand_landmarks, r.left_hand_world_landmarks),
                       (r.right_hand_landmarks, r.right_hand_world_landmarks)):
        if lms:
            img = np.array([(p.x, p.y, p.z) for p in lms], np.float32)
            wld = np.array([(p.x, p.y, p.z) for p in world], np.float32) if world else np.full((21, 3), np.nan, np.float32)
            found.append((img, wld))
    for slot, (img, wld) in assign_slots(found, out["pose"]).items():
        out["hands"][slot] = img
        out["hands_world"][slot] = wld
    return out


def assign_slots(found, pose):
    """Map detected hands to slot 0 (person's left) and 1 (right) by distance to the pose wrists."""
    if not found:
        return {}
    wrists = pose[list(POSE_WRIST), :2]
    if np.isnan(wrists).any():
        return {k: h for k, h in enumerate(found[:2])}
    cost = np.array([[np.linalg.norm(h[0][0, :2] - w) for w in wrists] for h in found])
    if len(found) == 1:
        return {int(cost[0].argmin()): found[0]}
    straight = cost[0, 0] + cost[1, 1]
    crossed = cost[0, 1] + cost[1, 0]
    return {0: found[0], 1: found[1]} if straight <= crossed else {0: found[1], 1: found[0]}


def stack_frames(frames):
    return {k: np.stack([f[k] for f in frames]) for k in frames[0]}


def clip_keypoints(holistic, frames, fps):
    step = 1000.0 / max(fps, 1.0)
    return stack_frames([frame_keypoints(holistic, f, int(i * step)) for i, f in enumerate(frames)])
