"""The shipped 2,000-word INT8 model scored on WLASL's standard sub-vocabularies (WLASL100 / 300 / 1000 / 2000):
the test clips of the N most common words, answers restricted to those N words, exactly as predict() does with
ECHO_VOCAB_FILE. Two ways of restricting are compared:

  mask   drop the words outside the list and renormalise: what predict() does with ECHO_VOCAB_FILE
  remap  inside each same-sign set, the probability of the names outside the list goes to the names inside it
  raw    the mask on the network's own probabilities, without the same-sign answer rule (ECHO_SET_RULE=0)

Also scores random word lists of each size (20 draws) with both, since the organizers' list need not be a WLASL
subset. Measured 15 Nov 2026 on the shipped model: remap is no better on the WLASL subsets and 0.3 to 0.8 points
worse on random lists (a set with a name outside the list takes probability from the true word more often than
it rescues its own), so predict() keeps the plain mask.

Usage: vocab_subsets.py <index.json> <cache.pt> <model_dir> [--out f.json]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_v2 as ex  # noqa: E402
import train_v2 as tv  # noqa: E402


def restrict(probs, keep, set_probs=None, set_of=None):
    """Probabilities when only the words `keep` may be answered: plain mask, or (with sets) the remap."""
    out = np.zeros_like(probs)
    if set_probs is None or set_of is None:
        out[keep] = probs[keep]
    else:
        inside = np.zeros(int(set_of.max()) + 1, dtype=np.float64)
        np.add.at(inside, set_of[keep], probs[keep])
        out[keep] = set_probs[keep] * probs[keep] / np.maximum(inside[set_of[keep]], 1e-12)
    return out / max(float(out.sum()), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("model_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    md = pathlib.Path(a.model_dir)
    labels = json.loads((md / "labels.json").read_text(encoding="utf-8"))
    n = len(labels)
    set_of = np.array(json.loads((md / "sets.json").read_text()))
    entries = json.load(open(a.index, encoding="utf-8"))
    cache = torch.load(a.cache, weights_only=False)
    ids, y = tv.select(cache, entries, n)["test"]
    feeds = ex.batches({k: cache[k] for k in ("X", "W", "S", "L")}, ids, json.loads((md / "export.json").read_text())["t_in"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(md / "sign_int8.onnx"), so, providers=["CPUExecutionProvider"])
    outs = [sess.run(["probs", "set_probs", "raw_probs"], f) for f in feeds]
    probs = np.concatenate([o[0] for o in outs])
    mass = np.concatenate([o[1] for o in outs])
    raw = np.concatenate([o[2] for o in outs])
    y = y.numpy()

    def score(keep, how):
        rows = np.isin(y, keep)
        hit1 = hit5 = 0
        for p, m, t in zip((raw if how == "raw" else probs)[rows], mass[rows], y[rows]):
            q = restrict(p, keep, m if how == "remap" else None, set_of if how == "remap" else None)
            order = np.argsort(-q)
            hit1 += order[0] == t
            hit5 += t in order[:5]
        k = int(rows.sum())
        return dict(clips=k, top1=round(100 * hit1 / k, 2), top5=round(100 * hit5 / k, 2))

    res = {}
    for size in (100, 300, 1000, 2000):
        keep = np.arange(size)
        res[f"wlasl{size}"] = dict(mask=score(keep, "mask"), remap=score(keep, "remap"), raw=score(keep, "raw"))
        print(f"WLASL{size}", res[f"wlasl{size}"], flush=True)
    rng = np.random.default_rng(0)
    for size in (100, 300, 1000):
        draws = [np.sort(rng.choice(n, size, replace=False)) for _ in range(20)]
        m = [score(k, "mask")["top1"] for k in draws]
        r = [score(k, "remap")["top1"] for k in draws]
        w = [score(k, "raw")["top1"] for k in draws]
        res[f"random{size}"] = dict(mask_top1_mean=round(float(np.mean(m)), 2), remap_top1_mean=round(float(np.mean(r)), 2),
                                    raw_top1_mean=round(float(np.mean(w)), 2), mask_better_than_raw_in=int(sum(b > c for b, c in zip(m, w))),
                                    remap_minus_mask_mean=round(float(np.mean(np.array(r) - np.array(m))), 2),
                                    remap_better_in=int(sum(b > c for b, c in zip(r, m))), draws=len(draws))
        print(f"random {size}", res[f"random{size}"], flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
