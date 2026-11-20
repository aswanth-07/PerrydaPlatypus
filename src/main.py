"""Track 2 entry point: predict(video_path) -> dict, as the brief specifies.

Pipeline (identical on the phone, see android/signcore): MediaPipe Holistic keypoints -> normalise.normalise
-> linear resample to the model's T steps -> the EchoGesture INT8 ONNX model (2,000 ASL words; dominant-hand
flip, features and the same-sign answer rule are inside the graph).

Usage: python src/main.py <video or folder> [--out predictions.jsonl] [--model-dir src/models]

Everything predict() needs lives in src/: this file, normalise.py, extract.py, holistic_kp.py and src/models/
(echogesture_int8.onnx, its labels and same-sign sets, and the MediaPipe Holistic task file).
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

import normalise as dv  # noqa: E402
import extract as ev  # noqa: E402
import holistic_kp as hk  # noqa: E402

# The EchoGesture model: all 2,000 WLASL words. ECHO_VOCAB_FILE restricts its answers to a published word list.
DEFAULT_MODEL_DIR = SRC / "models"
MODEL_FILE = "echogesture_int8.onnx"
# Every frame is landmarked, up to 96 (the cap the training keypoints were extracted with). On all 2,878 raw
# WLASL2000 test videos that scores 64.63% top-1 against 63.45% for 24 evenly spaced frames: Holistic follows the
# hands across every frame better than across wide gaps. A clip then costs about 57 Holistic calls instead of 24;
# ECHO_FRAMES=24 is the faster setting, and what the phone's Videos tab uses.
CLIP_FRAMES = int(os.environ.get("ECHO_FRAMES", "96"))
# Some WLASL words are one sign under several names (DOCTOR / PHYSICIAN): WLASL's own index lists the same video
# segment under both. A model exported with the same-sign answer rule adds up the probability of such a set and
# answers with the name WLASL uses most (scripts/signsets.py). ECHO_SET_RULE=0 answers from the network's own
# probabilities instead; it changes nothing for a model exported without the rule.
SET_RULE = os.environ.get("ECHO_SET_RULE", "1") != "0"


def _resample_weights(n, t_in):
    pos = np.linspace(0, 1, t_in) * max(n - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, max(n - 1, 0))
    w = (pos - lo).astype(np.float32)
    return lo, hi, w


def resample(a, t_in, pre=None):
    """Linear resampling along time, the same as train.sample_time at test time."""
    lo, hi, w = pre if pre is not None else _resample_weights(len(a), t_in)
    w = w.reshape((-1,) + (1,) * (a.ndim - 1))
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
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, threads=1, vocab_file=os.environ.get("ECHO_VOCAB_FILE"),
                 reuse_landmarker=os.environ.get("ECHO_FRESH_LANDMARKER") != "1"):
        model_dir = pathlib.Path(model_dir)
        self.models_dir = model_dir   # holistic_landmarker.task sits beside the model
        self.labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.t_in = json.loads((model_dir / "export.json").read_text())["t_in"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / MODEL_FILE), so, providers=["CPUExecutionProvider"])
        outputs = [o.name for o in self.session.get_outputs()]
        self.output = "raw_probs" if not SET_RULE and "raw_probs" in outputs else outputs[0]
        # ECHO_VOCAB_FILE: when the scored vocabulary is published, answer only from those words.
        self.keep = load_vocab_filter(self.labels, vocab_file) if vocab_file else None
        self.last_stages = {}
        # Up to 4 Holistic landmarker sets cached by (w, h), with per-size monotonic timestamps carried on;
        # a blank frame before each clip makes it lose the last clip's person so tracking starts afresh.
        # ECHO_FRESH_LANDMARKER=1 builds one per clip instead (about 450 ms slower per clip).
        self.reuse = reuse_landmarker
        from collections import OrderedDict
        self._hol_cache = OrderedDict()   # (w, h) -> [hol_0, ..., hol_{lanes-1}]
        self._hol_ts = {}                 # (w, h) -> next_ts_ms
        # ECHO_LANES=k (default 1): k landmarkers take the clip's frames in turn, in parallel threads, as the phone's
        # live mode does; 1 is the path every published number was measured with.
        self.lanes = max(1, int(os.environ.get("ECHO_LANES", "1")))
        self._pool = None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for hols in self._hol_cache.values():
            for h in hols:
                h.close()
        self._hol_cache.clear()
        self._hol_ts.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _get_hols(self, size):
        hols = self._hol_cache.get(size)
        if hols is not None:
            self._hol_cache.move_to_end(size)
            return hols
        while len(self._hol_cache) >= 4:
            old_size, old_hols = self._hol_cache.popitem(last=False)
            for h in old_hols:
                h.close()
            self._hol_ts.pop(old_size, None)
        hols = [hk.make_holistic(self.models_dir) for _ in range(self.lanes)]
        self._hol_cache[size] = hols
        self._hol_ts[size] = 0
        return hols

    def _keypoints(self, frames, fps):
        h, w = frames[0].shape[:2]
        if not self.reuse:
            with hk.make_holistic(self.models_dir) as hol:
                return hk.clip_keypoints(hol, frames, fps)
        size = (w, h)
        hols = self._get_hols(size)
        step = max(1.0, 1000.0 / max(fps, 1.0))  # guarantee strictly increasing ms timestamps even for >1000 fps metadata
        blank_ts = self._hol_ts[size]
        start = blank_ts + 1000
        self._hol_ts[size] = start + int(len(frames) * step) + len(frames) + 1000   # reserved first, so a clip that fails never reuses one
        blank = np.zeros_like(frames[0])
        if self.lanes > 1:
            return self._keypoints_lanes(hols, frames, blank, step, blank_ts, start)
        hk.frame_keypoints(hols[0], blank, blank_ts)
        out, last_ts = [], blank_ts
        for i, f in enumerate(frames):
            ts = max(start + int(i * step), last_ts + 1)
            last_ts = ts
            out.append(hk.frame_keypoints(hols[0], f, ts))
        return hk.stack_frames(out)

    def _keypoints_lanes(self, hols, frames, blank, step, blank_ts, start):
        from concurrent.futures import ThreadPoolExecutor
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.lanes)
        out: list = [None] * len(frames)

        def lane(k):
            hk.frame_keypoints(hols[k], blank, blank_ts)
            last_ts = blank_ts
            for i in range(k, len(frames), self.lanes):
                ts = max(start + int(i * step), last_ts + 1)
                last_ts = ts
                out[i] = hk.frame_keypoints(hols[k], frames[i], ts)

        list(self._pool.map(lane, range(self.lanes)))
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
            print(f"warning: no readable frames in {video_path}", file=sys.stderr)
            pts, wld, seen = (np.zeros((1, dv.P, 3), np.float32), np.zeros((1, 2, 21, 3), np.float32), np.zeros((1, 4), np.float32))
        t2 = time.perf_counter()
        pre = _resample_weights(len(pts), self.t_in)
        feed = dict(points=resample(pts, self.t_in, pre)[None].astype(np.float32),
                    world=resample(wld, self.t_in, pre)[None].astype(np.float32),
                    seen=resample(seen, self.t_in, pre)[None].astype(np.float32))
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
        res = dict(clip_id=pathlib.Path(video_path).stem, prediction=self.labels[order[0]].upper(),
                   top5=[self.labels[i].upper() for i in order[:5]], confidence=round(float(probs[order[0]]), 4),
                   latency_ms=round(1000 * (t3 - t2), 2),
                   pipeline_ms=round(1000 * (t3 - t0), 2),
                   keypoints_ms_per_frame=round(1000 * (t2 - t1) / max(len(frames), 1), 2))
        if not frames:
            res["no_frames"] = True
        return res


_default = None


def predict(video_path):
    global _default
    if _default is None:
        _default = Predictor()
    return _default(video_path)


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="predictions.jsonl")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    a = ap.parse_args()
    p = pathlib.Path(a.path)
    videos = sorted(v for v in p.iterdir() if v.is_file() and v.suffix.lower() in VIDEO_EXTS) if p.is_dir() else [p]
    with Predictor(a.model_dir) as pred, open(a.out, "w", encoding="utf-8") as f:
        for v in videos:
            r = pred(v)
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
