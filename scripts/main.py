"""Prototype of the required predict() entry point: video path in, one uppercase word out, all on the CPU."""
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as fx
import keypoints as kp

INPUT_T = 48


class Predictor:
    def __init__(self, model_path, labels, models_dir, frames=32):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.labels = labels
        self.models_dir = models_dir
        self.frames = frames
        self.last_stages = {}

    def __call__(self, video_path, transform=None):
        t0 = time.perf_counter()
        frames, fps = kp.read_frames(video_path, limit=self.frames)
        if transform is not None:
            frames = [transform(f) for f in frames]
        t1 = time.perf_counter()
        hands, pose = kp.make_detectors(self.models_dir)
        t1b = time.perf_counter()
        with hands, pose:
            pts = kp.clip_keypoints(hands, pose, frames, fps)
        t2 = time.perf_counter()
        h, w = frames[0].shape[:2]
        x, seen = fx.normalise_clip(pts, (w, h))
        inp = fx.make_features_np(fx.resample(x, INPUT_T).astype(np.float32),
                                  fx.resample(seen, INPUT_T).astype(np.float32))[None]
        t3 = time.perf_counter()
        logits = self.sess.run(None, {"features": inp})[0][0]
        t4 = time.perf_counter()
        order = np.argsort(-logits)[:5]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        self.last_stages = dict(frames=len(frames), decode_ms=1000 * (t1 - t0), load_ms=1000 * (t1b - t1),
                                keypoints_ms=1000 * (t2 - t1b), features_ms=1000 * (t3 - t2), classify_ms=1000 * (t4 - t3))
        return {
            "clip_id": pathlib.Path(video_path).stem,
            "prediction": self.labels[order[0]].upper(),
            "top5": [self.labels[i].upper() for i in order],
            "confidence": float(probs[order[0]]),
            "latency_ms": round(1000 * (t4 - t0) - self.last_stages["load_ms"], 1),
        }


def load_labels(index, classes):
    return [e["gloss"] for e in json.load(open(index, encoding="utf-8"))[:classes]]
