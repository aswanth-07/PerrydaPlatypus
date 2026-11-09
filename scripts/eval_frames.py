"""Accuracy when only N evenly spaced frames of each clip are landmarked (the rest never computed).

This approximates a faster predict(): Holistic runs on N frames instead of all of them. VIDEO-mode
tracking across larger gaps may behave slightly differently from this subsampling of the full run.

Usage: eval_frames.py <index.json> <cache.pt> <ckpt.pt> [<ckpt.pt> ...] --subset 100 [--frames 8,12,16,24,32,48]
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_v2 as mv
import train_v2 as tv


def subsample(data, ids, n):
    X, W, S, L = (data[k][ids] for k in ("X", "W", "S", "L"))
    m = torch.minimum(L, torch.tensor(n, device=L.device))
    pos = torch.linspace(0, 1, n, device=L.device)[None] * (L[:, None] - 1).clamp_min(0).float()
    idx = pos.round().long()
    r = torch.arange(len(ids), device=L.device)[:, None]
    return dict(X=X[r, idx], W=W[r, idx], S=S[r, idx], L=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--frames", default="8,12,16,24,32,48")
    a = ap.parse_args()
    dev = "cuda"
    entries = json.load(open(a.index, encoding="utf-8"))
    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, a.subset)
    data = {k: cache[k].to(dev) for k in ("X", "W", "S", "L")}
    models = []
    for c in a.ckpts:
        ck = torch.load(c, weights_only=False)
        m = mv.build(ck["cfg"], ck["classes"]).to(dev)
        m.load_state_dict(ck["state"])
        models.append((m.eval(), ck["cfg"]))
    perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).to(dev)
    ids, y = rows["test"]
    ids, y = ids.to(dev), y.to(dev)
    res = {}
    full = sum(tv.predict_logits(m, data, ids, cfg, False, perm) for m, cfg in models)
    res["all"] = tv.score(full, y)["top1"]
    for n in map(int, a.frames.split(",")):
        sub = subsample(data, ids, n)
        probs = sum(tv.predict_logits(m, sub, torch.arange(len(ids), device=dev), cfg, False, perm) for m, cfg in models)
        res[str(n)] = tv.score(probs, y)["top1"]
    res["mean_frames_all"] = round(float(data["L"][ids].float().mean()), 1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
