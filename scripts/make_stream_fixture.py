"""Expected live captions for the Kotlin SpotterTest: the clips of android/parity/fixture.json chained into streams
and captioned by the Python boundary spotter (live_eval.run_stream_seg) with the INT8 model in that folder.
Spotter.kt must give the same words at the same frames.

Each stream is `per` consecutive fixture clips at 25 fps (40 ms a frame) followed by half a second with nobody in
view, with the first clip's frame size for every frame. Writes android/parity/streams.json (a few kilobytes).

Usage: make_stream_fixture.py <parity_dir> [--per 6]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import live_eval as le

FPS = 25.0


def arr(v, shape):
    return np.full(shape, np.nan) if v is None else np.asarray(v, np.float64).reshape(shape)


def raw_frames(clip):
    return [dict(pose=arr(f["pose"], (33, 4)), hands=np.stack([arr(h, (21, 3)) for h in f["hands"]]),
                 face=arr(f["face"], (62, 3)), hands_world=np.stack([arr(h, (21, 3)) for h in f["world"]]))
            for f in clip["frames"]]


class OnnxClassifier:
    """The exported INT8 graph, as the phone runs it: resample to t_in, read `probs` and `set_probs`."""

    def __init__(self, path, t_in):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.t_in = t_in

    def __call__(self, pts, wld, seen):
        def rs(a):
            n = len(a)
            pos = np.linspace(0, 1, self.t_in) * max(n - 1, 0)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, max(n - 1, 0))
            w = (pos - lo).astype(np.float32).reshape((-1,) + (1,) * (a.ndim - 1))
            return (a[lo] + (a[hi] - a[lo]) * w)[None].astype(np.float32)

        p, mass = self.sess.run(["probs", "set_probs"], dict(points=rs(pts), world=rs(wld), seen=rs(seen)))
        return p[0], mass[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parity_dir")
    ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()
    d = pathlib.Path(a.parity_dir)
    fixture = json.loads((d / "fixture.json").read_text())
    labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    labels = [l.upper() for l in labels]
    clf = OnnxClassifier(d / "sign_int8.onnx", fixture["t_in"])
    clips = fixture["clips"]
    streams = []
    for s in range(0, len(clips) - a.per + 1, a.per):
        ids = list(range(s, s + a.per))
        frames = [f for i in ids for f in raw_frames(clips[i])]
        wh = (clips[s]["width"], clips[s]["height"])
        rest = int(round(FPS / 2))
        cap = le.run_stream_seg(le.rest_frames(frames, rest), wh, FPS, clf, labels)
        streams.append(dict(clips=ids, truth=[clips[i]["label"] for i in ids], rest_frames=rest,
                            caption=[[w, int(t)] for w, t in cap]))
        print(ids, [clips[i]["label"] for i in ids], "->", cap)
    (d / "streams.json").write_text(json.dumps(dict(fps=FPS, streams=streams), indent=1))


if __name__ == "__main__":
    main()
