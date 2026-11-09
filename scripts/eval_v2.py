"""Score one or more SignNet checkpoints (an ensemble if several) on the test split, clean and under each
degraded condition, on exactly the same clips. Accuracy drop is clean top-1 minus condition top-1.

Usage: eval_v2.py <index.json> <clean_kp_dir> <robust_dir> <ckpt.pt> [<ckpt.pt> ...] [--subset 100] [--tta] [--out file.json]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_v2 as dv
import model_v2 as mv
import train_v2 as tv


def load_dir(kp_dir, vids):
    files = [pathlib.Path(kp_dir) / f"{v}.npz" for v in vids]
    n = len(files)
    X = np.zeros((n, dv.T_MAX, dv.P, 3), np.float32)
    W = np.zeros((n, dv.T_MAX, 2, 21, 3), np.float32)
    S = np.zeros((n, dv.T_MAX, 4), np.float32)
    L = np.zeros(n, np.int64)
    for k, f in enumerate(files):
        pts, wld, seen = dv.load_clip(f, "holistic")
        t = min(len(pts), dv.T_MAX)
        X[k, :t], W[k, :t], S[k, :t], L[k] = pts[:t], wld[:t], seen[:t], t
    return {k: torch.from_numpy(v).cuda() for k, v in dict(X=X, W=W, S=S, L=L).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("clean")
    ap.add_argument("robust")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    entries = json.load(open(a.index, encoding="utf-8"))[:a.subset]
    conds = sorted(p.name for p in pathlib.Path(a.robust).iterdir() if p.is_dir())
    vids, labels = [], []
    for cls, e in enumerate(entries):
        for i in e["instances"]:
            v = i["video_id"]
            if i["split"] == a.split and (pathlib.Path(a.clean) / f"{v}.npz").exists() and \
                    all((pathlib.Path(a.robust) / c / f"{v}.npz").exists() for c in conds):
                vids.append(v)
                labels.append(cls)
    y = torch.tensor(labels).cuda()
    models = []
    for c in a.ckpts:
        ck = torch.load(c, weights_only=False)
        m = mv.build(ck["cfg"], ck["classes"]).cuda()
        m.load_state_dict(ck["state"])
        models.append((m.eval(), ck["cfg"]))
    perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).cuda()
    res = {}
    for cond, d in [("clean", a.clean)] + [(c, pathlib.Path(a.robust) / c) for c in conds]:
        data = load_dir(d, vids)
        ids = torch.arange(len(vids)).cuda()
        probs = sum(tv.predict_logits(m, data, ids, cfg, a.tta, perm) for m, cfg in models)
        res[cond] = tv.score(probs, y)
    clean = res["clean"]["top1"]
    for cond in res:
        res[cond]["drop"] = round(clean - res[cond]["top1"], 2)
    print(f"{len(vids)} clips, {len(models)} model(s), tta={a.tta}")
    for cond, r in res.items():
        print(f"  {cond:12s} top1 {r['top1']:5.1f}  top5 {r['top5']:5.1f}  drop {r['drop']:+5.1f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(clips=len(vids), ckpts=a.ckpts, tta=a.tta, results=res), indent=1))


if __name__ == "__main__":
    main()
