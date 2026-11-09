"""Phonological re-ranking: score(sign) = log p(sign) + beta * mean_t log p_t(phoneme_t(sign)), using the
auxiliary phoneme heads a --phonology checkpoint saved. Signs without ASL-LEX labels keep their plain score
(their phoneme term is the mean over labelled signs, so they are neither helped nor hurt on average).
beta is chosen on validation; test is reported for the chosen beta only.

Usage: rerank_phon.py <index.json> <cache.pt> <signdata.csv> <ckpt.pt> --subset N
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_v2 as mv
import phonology as ph
import train_v2 as tv


@torch.no_grad()
def scores(model, aux, data, ids, cfg, perm, labels):
    model.eval()
    sign, phon = [], []
    for chunk in ids.split(512):
        X, W, S, L = (data[k][chunk] for k in ("X", "W", "S", "L"))
        x, w, s = tv.sample_time(X, W, S, L, cfg["t_in"], False)
        if cfg["canon"]:
            x, w, s = tv.apply_flip(x, w, s, tv.dominant_left(x, s, torch.full_like(L, cfg["t_in"]), cfg.get("canon_rel", False)), perm)
        emb = model.embed(x, w, s)
        sign.append(torch.log_softmax(model.head(emb), -1))
        terms = []
        for t, head in aux.items():
            lp = torch.log_softmax(head(emb), -1)                      # B x values
            lab = labels[t]                                            # classes, -1 unknown
            per_class = lp[:, lab.clamp_min(0)]                        # B x classes
            known = (lab >= 0).float()
            fill = (per_class * known).sum(1, keepdim=True) / known.sum().clamp_min(1)
            terms.append(torch.where(lab[None] >= 0, per_class, fill))
        phon.append(torch.stack(terms).mean(0))
    return torch.cat(sign), torch.cat(phon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("signdata")
    ap.add_argument("ckpt")
    ap.add_argument("--subset", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda"
    entries = json.load(open(a.index, encoding="utf-8"))
    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, a.subset)
    data = {k: cache[k].to(dev) for k in ("X", "W", "S", "L")}
    ck = torch.load(a.ckpt, weights_only=False)
    model = mv.build(ck["cfg"], ck["classes"]).to(dev)
    model.load_state_dict(ck["state"])
    lab, sizes, _ = ph.load(a.signdata, [e["gloss"] for e in entries[:a.subset]])
    aux = torch.nn.ModuleDict({t: torch.nn.Linear(2 * ck["cfg"]["dim"], n) for t, n in sizes.items()}).to(dev)
    aux.load_state_dict(ck["aux"])
    labels = {t: torch.tensor(v, device=dev) for t, v in lab.items()}
    perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).to(dev)
    out = {}
    for split in ("val", "test"):
        i, y = rows[split]
        out[split] = (scores(model, aux, data, i.to(dev), ck["cfg"], perm, labels), y.to(dev))
    (sv, pv), yv = out["val"]
    best = max((float((torch.argmax(sv + b * pv, 1) == yv).float().mean()), b) for b in (0, 0.25, 0.5, 1, 2, 3, 5))
    (st, pt), yt = out["test"]
    res = {f"beta_{b}": round(100 * float((torch.argmax(st + b * pt, 1) == yt).float().mean()), 2) for b in (0, best[1])}
    res["chosen_beta_on_val"] = best[1]
    res["val_top1_at_beta"] = round(100 * best[0], 2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
