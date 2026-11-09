"""Label-shift correction: does reweighting by the WLASL class prior help on a WLASL split?

ASL Citizen adds about 30 clips to nearly every word, so the training mixture's class prior is much flatter than
WLASL's long tail, while WLASL's val and test splits follow WLASL's own frequencies. The correction multiplies
each class probability by (p_wlasl(c) / p_mix(c)) ** alpha, one constant vector computed from training counts only.

Usage: prior_eval.py <ckpt.pt> [<ckpt.pt> ...] --split val|test [--manifest data/aslcitizen/manifest_trainval.json]
       [--trainval] [--alphas 0,0.25,0.5,0.75,1] [--out prior.json]
The prior vector is written with --out (log ratio per class) so predict() and the exported graph can use it.
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_v2 as mv  # noqa: E402
import train_v2 as tv  # noqa: E402

N = 2000


def log_ratio(entries, manifest, trainval, aug_per_train=1.0):
    wl = np.ones(N)          # +1 smoothing
    mix = np.ones(N)
    splits = ("train", "val") if trainval else ("train",)
    for c, e in enumerate(entries[:N]):
        for i in e["instances"]:
            if i["split"] == "train":
                wl[c] += 1
                mix[c] += 1 + aug_per_train
            elif i["split"] in splits:
                mix[c] += 1
    for row in json.load(open(manifest, encoding="utf-8")):
        if row["cls"] < N:
            mix[row["cls"]] += 1
    p_wl, p_mix = wl / wl.sum(), mix / mix.sum()
    return np.log(p_wl) - np.log(p_mix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--index", default="data/index/WLASL_v0.3.json")
    ap.add_argument("--cache", default="data/cache_hol2000w.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--manifest", default="data/aslcitizen/manifest_trainval.json")
    ap.add_argument("--trainval", action="store_true", help="the models trained on WLASL train+val (mixture counts)")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.25")
    ap.add_argument("--out")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))
    r = log_ratio(entries, a.manifest, a.trainval)
    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, N)
    ids, y = rows[a.split]
    data = {k: cache[k][ids].cuda() for k in ("X", "W", "S", "L")}   # only the scored clips go to the GPU
    del cache
    ids = torch.arange(len(y))
    probs, ck = 0, None
    for c in a.ckpts:
        ck = torch.load(c, weights_only=False)
        m = mv.build(ck["cfg"], ck["classes"]).cuda()
        m.load_state_dict(ck["state"])
        perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).cuda()
        probs = probs + tv.predict_logits(m.eval(), data, ids.cuda(), ck["cfg"], False, perm)[:, :N]
    probs = probs.cpu().double()
    y = y.long()
    rt = torch.from_numpy(r)
    res = {}
    for al in map(float, a.alphas.split(",")):
        p = probs * torch.exp(al * rt)[None]
        top = p.argsort(1, descending=True)[:, :5]
        res[al] = dict(top1=round(100 * float((top[:, 0] == y).double().mean()), 2),
                       top5=round(100 * float((top == y[:, None]).any(1).double().mean()), 2))
        print(f"alpha {al:4.2f}  {a.split} top1 {res[al]['top1']:5.2f}  top5 {res[al]['top5']:5.2f}", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(log_ratio=[float(v) for v in r], split=a.split, ckpts=a.ckpts,
                                                       results={str(k): v for k, v in res.items()}), indent=0))


if __name__ == "__main__":
    main()
