"""Track 2 entry point: predict(video_path) -> dict, as the brief specifies.

Pipeline (identical on the phone, see android/signcore): MediaPipe Holistic keypoints -> data_v2.normalise
-> linear resample to the model's T steps -> INT8 ONNX classifier (dominant-hand flip, features and the
ensemble average are inside the graph).

Usage: python src/main.py <video or folder> [--out predictions.jsonl] [--model-dir src/models/full] [--models-dir DIR]

Everything predict() needs lives in src/: this file, data_v2.py, extract_v2.py, holistic_kp.py and src/models/
(INT8 classifiers and the MediaPipe Holistic task file).
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "3")
SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

import data_v2 as dv  # noqa: E402
import extract_v2 as ev  # noqa: E402
import holistic_kp as hk  # noqa: E402

# src/models/full: all 2,000 WLASL words (default, since the scored vocabulary is not published);
# src/models/core: the 100 most common words, most accurate; src/models/union: 3,075 words (WLASL + ASL Citizen).
# ECHO_VOCAB=core|full|union picks one; ECHO_MODELS points at another folder of model directories.
MODELS = pathlib.Path(os.environ.get("ECHO_MODELS", SRC / "models"))
DEFAULT_MODEL_DIR = MODELS / os.environ.get("ECHO_VOCAB", "full")
# Landmarking 24 evenly spaced frames scored the same as all frames on the WLASL100 test split (84.5 vs 84.5,
# real extraction; 16 frames: 83.7), so a clip costs about 24 Holistic calls instead of ~58.
CLIP_FRAMES = int(os.environ.get("ECHO_FRAMES", "24"))
# Some WLASL words are one sign under several names (DOCTOR / PHYSICIAN): WLASL's own index lists the same video
# segment under both. A model exported with the same-sign answer rule adds up the probability of such a set and
# answers with the name WLASL uses most (scripts/signsets.py). ECHO_SET_RULE=0 answers from the network's own
# probabilities instead; it changes nothing for a model exported without the rule.
SET_RULE = os.environ.get("ECHO_SET_RULE", "1") != "0"


def resample(a, t_in):
    """Linear resampling along time, the same as train_v2.sample_time at test time."""
    n = len(a)
    pos = np.linspace(0, 1, t_in) * max(n - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, max(n - 1, 0))
    w = (pos - lo).astype(np.float32).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[lo] + (a[hi] - a[lo]) * w


def load_vocab_filter(labels, path):
    """Allowed words, one per line or a JSON list; returns label indices to keep (unknown words are reported)."""
    text = pathlib.Path(path).read_text(encoding="utf-8").strip()
    words = json.loads(text) if text.startswith("[") else [w for w in text.splitlines() if w.strip()]
    index = {l.upper(): i for i, l in enumerate(labels)}
    keep = sorted({index[w.strip().upper()] for w in words if w.strip().upper() in index})
    missing = [w for w in words if w.strip().upper() not in index]
    if missing:
        print(f"vocabulary filter: {len(missing)} word(s) not in the model's vocabulary, e.g. {missing[:5]}", file=sys.stderr)
    return np.array(keep, dtype=int)


class Predictor:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, models_dir=None, threads=1, vocab_file=os.environ.get("ECHO_VOCAB_FILE"),
                 reuse_landmarker=os.environ.get("ECHO_FRESH_LANDMARKER") != "1"):
        model_dir = pathlib.Path(model_dir)
        if models_dir is None:  # holistic_landmarker.task sits beside the vocabulary folders
            models_dir = model_dir if (model_dir / "holistic_landmarker.task").exists() else model_dir.parent
        self.models_dir = pathlib.Path(models_dir)
        self.labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.t_in = json.loads((model_dir / "export.json").read_text())["t_in"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "sign_int8.onnx"), so, providers=["CPUExecutionProvider"])
        outputs = [o.name for o in self.session.get_outputs()]
        self.output = "raw_probs" if not SET_RULE and "raw_probs" in outputs else outputs[0]
        # ECHO_VOCAB_FILE: when the scored vocabulary is published, answer only from those words.
        self.keep = load_vocab_filter(self.labels, vocab_file) if vocab_file else None
        self.last_stages = {}
        # One Holistic landmarker kept across clips (rebuilt when the frame size changes), with timestamps carried
        # on; a blank frame before each clip makes it lose the last clip's person so tracking starts afresh.
        # ECHO_FRESH_LANDMARKER=1 builds one per clip instead (about 450 ms slower per clip).
        self.reuse = reuse_landmarker
        self._hol, self._size, self._ts = None, None, 0
        # ECHO_LANES=k (default 1): k landmarkers take the clip's frames in turn, in parallel threads, as the phone's
        # live mode does; 1 is the path every published number was measured with.
        self.lanes = max(1, int(os.environ.get("ECHO_LANES", "1")))
        self._extra = []

    def _keypoints(self, frames, fps):
        h, w = frames[0].shape[:2]
        if not self.reuse:
            with hk.make_holistic(self.models_dir) as hol:
                return hk.clip_keypoints(hol, frames, fps)
        if self._hol is None or self._size != (w, h):
            if self._hol is not None:
                self._hol.close()
            self._hol, self._size = hk.make_holistic(self.models_dir), (w, h)
        step = 1000.0 / max(fps, 1.0)
        blank_ts, start = self._ts, self._ts + 1000
        self._ts = start + int(len(frames) * step) + 1000   # reserved first, so a clip that fails never reuses one
        if self.lanes > 1:
            return self._keypoints_lanes(frames, step, blank_ts, start)
        hk.frame_keypoints(self._hol, np.zeros_like(frames[0]), blank_ts)
        return hk.stack_frames([hk.frame_keypoints(self._hol, f, start + int(i * step)) for i, f in enumerate(frames)])

    def _keypoints_lanes(self, frames, step, blank_ts, start):
        from concurrent.futures import ThreadPoolExecutor
        size = frames[0].shape[:2]
        if len(self._extra) != self.lanes - 1 or getattr(self, "_extra_size", None) != size:
            for h in self._extra:
                h.close()
            self._extra = [hk.make_holistic(self.models_dir) for _ in range(self.lanes - 1)]
            self._extra_size = size
        hols = [self._hol] + self._extra
        out: list = [None] * len(frames)

        def lane(k):
            hk.frame_keypoints(hols[k], np.zeros_like(frames[0]), blank_ts)
            for i in range(k, len(frames), self.lanes):
                out[i] = hk.frame_keypoints(hols[k], frames[i], start + int(i * step))

        with ThreadPoolExecutor(self.lanes) as pool:
            list(pool.map(lane, range(self.lanes)))
        return hk.stack_frames(out)

    def __call__(self, video_path, transform=None):
        t0 = time.perf_counter()
        frames, fps = ev.read_frames(video_path, None, CLIP_FRAMES)
        if transform is not None:
            frames = transform(frames)
        t1 = time.perf_counter()
        if frames:
            raw = self._keypoints(frames, fps)
            h, w = frames[0].shape[:2]
            pts, wld, seen = dv.normalise(raw["pose"].astype(np.float64), raw["hands"].astype(np.float64),
                                          raw["face"].astype(np.float64), raw["hands_world"].astype(np.float64), (w, h))
        else:
            pts, wld, seen = (np.zeros((1, dv.P, 3), np.float32), np.zeros((1, 2, 21, 3), np.float32), np.zeros((1, 4), np.float32))
        t2 = time.perf_counter()
        feed = dict(points=resample(pts, self.t_in)[None].astype(np.float32), world=resample(wld, self.t_in)[None].astype(np.float32),
                    seen=resample(seen, self.t_in)[None].astype(np.float32))
        probs = self.session.run([self.output], feed)[0][0]
        t3 = time.perf_counter()
        if self.keep is not None and len(self.keep):
            masked = np.zeros_like(probs)
            masked[self.keep] = probs[self.keep]
            probs = masked / max(float(masked.sum()), 1e-12)
        order = np.argsort(-probs)
        self.last_stages = dict(decode_ms=round(1000 * (t1 - t0), 2), keypoints_ms=round(1000 * (t2 - t1), 2),
                                classifier_ms=round(1000 * (t3 - t2), 3), frames=len(frames))
        # latency_ms is the quantized model's inference time, the figure the brief's <= 50 ms budget is about.
        # The whole clip (decode + MediaPipe Holistic on every landmarked frame + model) is reported beside it.
        return dict(clip_id=pathlib.Path(video_path).stem, prediction=self.labels[order[0]].upper(),
                    top5=[self.labels[i].upper() for i in order[:5]], confidence=round(float(probs[order[0]]), 4),
                    latency_ms=round(1000 * (t3 - t2), 2),
                    pipeline_ms=round(1000 * (t3 - t0), 2),
                    keypoints_ms_per_frame=round(1000 * (t2 - t1) / max(len(frames), 1), 2))


_default = None


def predict(video_path):
    global _default
    if _default is None:
        _default = Predictor()
    return _default(video_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="predictions.jsonl")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--models-dir", default=None, help="folder holding holistic_landmarker.task (default: beside the model)")
    a = ap.parse_args()
    p = pathlib.Path(a.path)
    videos = sorted(p.glob("*.mp4")) if p.is_dir() else [p]
    pred = Predictor(a.model_dir, a.models_dir)
    with open(a.out, "w", encoding="utf-8") as f:
        for v in videos:
            r = pred(v)
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
