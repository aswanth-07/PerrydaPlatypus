"""Expected live captions for the Kotlin SpotterTest: the clips of android/parity/fixture.json chained into streams
and captioned in Python with the INT8 models the phone runs. Spotter.kt must give the same words at the same frames.

Two spotters are recorded for every stream:
  caption     the boundary spotter (live_eval.run_stream_seg) with the clip model (src/models)
  continuous  the continuous spotter (live_cont.run_cont: one caption per run of sign windows) with the live model given by --live,
              its decoder settings read from the live.json beside it; left out when --live is not given

Each stream is `per` consecutive fixture clips at 25 fps (40 ms a frame) followed by half a second with nobody in
view, with the first clip's frame size for every frame. Writes android/parity/streams.json (a few kilobytes).

Usage: make_stream_fixture.py <parity_dir> [--per 6] [--model-dir src/models] [--live android/models/echogesture_live_int8.onnx --sets src/models/sets.json]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
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
    ap.add_argument("--model-dir", default=str(pathlib.Path(__file__).resolve().parents[1] / "src" / "models"))
    ap.add_argument("--live", help="the live INT8 model (echogesture_live_int8.onnx, with live.json beside it)")
    ap.add_argument("--sets", help="sets.json of the vocabulary: words that are one sign gather evidence together")
    a = ap.parse_args()
    d = pathlib.Path(a.parity_dir)
    fixture = json.loads((d / "fixture.json").read_text())
    md = pathlib.Path(a.model_dir)
    labels = json.loads((md / "labels.json").read_text(encoding="utf-8"))
    labels = [l.upper() for l in labels]
    clf = OnnxClassifier(md / "echogesture_int8.onnx", fixture["t_in"])
    live = cfg = set_of = None
    if a.live:
        import live_cont as lc
        live = lc.OnnxWindowModel(a.live, fixture["t_in"], half=False)
        cfg = json.loads((pathlib.Path(a.live).parent / "live.json").read_text())
        set_of = np.array(json.loads(pathlib.Path(a.sets).read_text())) if a.sets else None
    clips = fixture["clips"]
    streams = []
    for s in range(0, len(clips) - a.per + 1, a.per):
        ids = list(range(s, s + a.per))
        frames = [f for i in ids for f in raw_frames(clips[i])]
        wh = (clips[s]["width"], clips[s]["height"])
        rest = int(round(FPS / 2))
        frames = le.rest_frames(frames, rest)
        cap = le.run_stream_seg(frames, wh, FPS, clf, labels)
        row = dict(clips=ids, truth=[clips[i]["label"] for i in ids], rest_frames=rest,
                   caption=[[w, int(t)] for w, t in cap])
        print(ids, row["truth"], "->", cap)
        if live is not None:
            post, hs = lc.posteriors(frames, wh, FPS, live, cfg["window_ms"] / 1000.0)
            cont = lc.run_cont(len(frames), FPS, post, hs, labels, set_of, stride=int(round(cfg["stride_ms"] * FPS / 1000.0)),
                               lo=cfg["lo"], gap=cfg["gap"], emin=cfg["emin"], commit=cfg["commit"], tail_ms=cfg["tail_ms"])
            row["continuous"] = [[w, int(t)] for w, t in cont]
            print("   continuous ->", cont)
        streams.append(row)
    out = dict(fps=FPS, streams=streams)
    if cfg is not None:
        out["live"] = {k: cfg[k] for k in ("window_ms", "stride_ms", "lo", "gap", "emin", "commit", "tail_ms")}
    (d / "streams.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
