"""Write the fixture the Kotlin ParityTest checks against: raw Holistic frames of some test clips, the
normalised and resampled model inputs Python computes from them, and the INT8 model's probabilities.

Usage: make_parity_fixture.py <index.json> <kp_dir> <export_dir> <out_dir> [--subset 100] [--clips 40]
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_v2 as dv
import train_v2 as tv


def or_null(a):
    a = np.asarray(a, np.float32)
    return None if not np.isfinite(a).all() else [float(v) for v in a.reshape(-1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("export_dir")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--clips", type=int, default=40)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = pathlib.Path(a.export_dir)
    t_in = json.loads((exp / "export.json").read_text())["t_in"]
    sess = ort.InferenceSession(str(exp / "sign_int8.onnx"), providers=["CPUExecutionProvider"])
    entries = json.load(open(a.index, encoding="utf-8"))[:a.subset]
    tests = [(i["video_id"], e["gloss"].upper()) for e in entries for i in e["instances"]
             if i["split"] == "test" and (pathlib.Path(a.kp_dir) / f"{i['video_id']}.npz").exists()]
    clips = []
    for vid, label in tests[:a.clips]:
        f = pathlib.Path(a.kp_dir) / f"{vid}.npz"
        d = np.load(f)
        frames = []
        for t in range(len(d["pose"])):
            hands = [or_null(d["hands"][t, s]) for s in (0, 1)]
            frames.append(dict(pose=or_null(d["pose"][t]), hands=hands,
                               world=[or_null(d["hands_world"][t, s]) if hands[s] is not None else None for s in (0, 1)],
                               face=or_null(d["face"][t])))
        pts, wld, seen = dv.load_clip(f, "holistic")
        L = torch.tensor([len(pts)])
        x, w, s = tv.sample_time(torch.from_numpy(pts)[None], torch.from_numpy(wld)[None], torch.from_numpy(seen)[None], L, t_in, False)
        feed = dict(points=x.numpy(), world=w.numpy(), seen=s.numpy())
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
        probs = sess.run(None, feed)[0][0]
        raw = outs.get("raw_probs")   # the network's own probabilities, before the same-sign answer rule
        clips.append(dict(video_id=vid, label=label, width=int(d["wh"][0]), height=int(d["wh"][1]), frames=frames,
                          points=[float(v) for v in feed["points"].reshape(-1)],
                          world_in=[float(v) for v in feed["world"].reshape(-1)],
                          seen=[float(v) for v in feed["seen"].reshape(-1)],
                          probs=[float(v) for v in probs],
                          **({} if raw is None else dict(raw_probs=[float(v) for v in raw[0]]))))
    (out / "fixture.json").write_text(json.dumps(dict(t_in=t_in, clips=clips)), encoding="utf-8")
    shutil.copy(exp / "sign_int8.onnx", out / "sign_int8.onnx")
    shutil.copy(exp / "labels.json", out / "labels.json")
    print(f"{len(clips)} clips -> {out / 'fixture.json'} ({(out / 'fixture.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
