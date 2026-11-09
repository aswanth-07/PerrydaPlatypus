"""Score checkpoints (an ensemble if several) on ASL Citizen's test split: 11 signers that appear in neither
ASL Citizen's train/val split nor WLASL, recorded on their own webcams. This is the unseen-signer estimate.

Usage: eval_aslc.py <aslc_cache.pt> <manifest_test.json> <ckpt.pt> [<ckpt.pt> ...] [--subset 100] [--out f.json]
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_v2 as mv
import train_v2 as tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("manifest")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--out")
    a = ap.parse_args()
    cache = torch.load(a.cache, weights_only=False)
    cls = {"aslc_" + pathlib.Path(c["file"]).stem: c["cls"] for c in json.load(open(a.manifest, encoding="utf-8"))}
    rows = [(n, cls[v]) for n, v in enumerate(cache["vid"]) if cls.get(v, a.subset) < a.subset]
    ids = torch.tensor([r[0] for r in rows]).cuda()
    y = torch.tensor([r[1] for r in rows]).cuda()
    data = {k: cache[k].cuda() for k in ("X", "W", "S", "L")}
    probs = 0
    for c in a.ckpts:
        ck = torch.load(c, weights_only=False)
        m = mv.build(ck["cfg"], ck["classes"]).cuda()
        m.load_state_dict(ck["state"])
        perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).cuda()
        probs = probs + tv.predict_logits(m.eval(), data, ids, ck["cfg"], False, perm)[:, :a.subset]
    res = tv.score(probs, y)
    print(json.dumps(dict(clips=len(rows), signers="unseen (ASL Citizen test split)", ckpts=len(a.ckpts), **res)))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
