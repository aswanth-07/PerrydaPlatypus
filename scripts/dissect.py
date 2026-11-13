"""Dissect the shipped 2,000-word model on the WLASL test split: where the errors are.

Reads only; writes one JSON report. Usage: dissect.py <repo> <ckpt> <out.json>
"""
import collections
import json
import pathlib
import sys

import numpy as np
import torch

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "scripts"))
import data_v2 as dv  # noqa: E402
import model_v2 as mv  # noqa: E402
import train_v2 as tv  # noqa: E402

dev = "cuda"
entries = json.load(open(repo / "data/index/WLASL_v0.3.json", encoding="utf-8"))[:2000]
rep = {}

# 1. Sign variants the index already labels (variation_id), which training ignores.
multi, test_in_multi, n_test, fine = 0, 0, 0, 0
var_sizes = []
for e in entries:
    by_var = collections.Counter(i["variation_id"] for i in e["instances"] if i["split"] != "test")
    big = [v for v, c in by_var.items() if c >= 2]
    fine += max(len(big), 1)
    t = sum(1 for i in e["instances"] if i["split"] == "test")
    n_test += t
    if len(big) >= 2:
        multi += 1
        test_in_multi += t
    var_sizes.append(len(by_var))
rep["variants"] = dict(glosses_with_2plus_variants_of_2plus_clips=multi, share_of_test_clips_in_them=round(100 * test_in_multi / n_test, 1),
                       fine_classes=fine, mean_variants_per_gloss=round(float(np.mean(var_sizes)), 2),
                       hist={str(k): int(v) for k, v in sorted(collections.Counter(var_sizes).items())})

# 2. The shipped float checkpoint on the test split.
cache = torch.load(repo / "data/cache_hol2000w.pt", weights_only=False)
rows = tv.select(cache, entries, 2000)
ids, y = rows["test"]
ck = torch.load(sys.argv[2], weights_only=False)
cfg = ck["cfg"]
model = mv.build(cfg, ck["classes"]).to(dev)
model.load_state_dict(ck["state"])
data = {k: cache[k][ids].to(dev) for k in ("X", "W", "S", "L")}
perm = tv.flip_perm(cache["face_flip"]).to(dev)
local = torch.arange(len(ids), device=dev)
probs = tv.predict_logits(model, data, local, cfg, False, perm).cpu()
y = y.cpu()
order = probs.argsort(1, descending=True)
rank = (order == y[:, None]).float().argmax(1)
hit = rank == 0
rep["test"] = dict(n=len(y), top1=round(100 * hit.float().mean().item(), 2),
                   top2=round(100 * (rank < 2).float().mean().item(), 2), top3=round(100 * (rank < 3).float().mean().item(), 2),
                   top5=round(100 * (rank < 5).float().mean().item(), 2), top10=round(100 * (rank < 10).float().mean().item(), 2))

# 3. Accuracy against how often the landmarker saw the hands.
S, L = cache["S"][ids].float(), cache["L"][ids]
valid = (torch.arange(S.shape[1])[None] < L[:, None]).float()
seen = (S * valid[..., None]).sum(1) / L[:, None].clamp_min(1)          # per clip: pose, left, right, face
best_hand = torch.maximum(seen[:, 1], seen[:, 2])
both = torch.minimum(seen[:, 1], seen[:, 2])
rep["seen_rate_mean"] = dict(pose=round(seen[:, 0].mean().item(), 3), better_hand=round(best_hand.mean().item(), 3),
                             worse_hand=round(both.mean().item(), 3), face=round(seen[:, 3].mean().item(), 3))
bins = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
rep["top1_by_better_hand_seen"] = {f"{a}-{b}": dict(n=int(m.sum()), top1=round(100 * hit[m].float().mean().item(), 1) if m.any() else None)
                                   for a, b in bins for m in [(best_hand >= a) & (best_hand < b)]}

# 4. Where the wrong answers go: confidence, margin, and symmetric confusion pairs (near-identical signs).
p1 = probs.max(1).values
rep["confidence"] = dict(mean_when_right=round(p1[hit].mean().item(), 3), mean_when_wrong=round(p1[~hit].mean().item(), 3))
wrong_rank = rank[~hit]
rep["wrong_by_rank"] = {"2": int((wrong_rank == 1).sum()), "3-5": int(((wrong_rank >= 2) & (wrong_rank < 5)).sum()),
                        "6-20": int(((wrong_rank >= 5) & (wrong_rank < 20)).sum()), ">20": int((wrong_rank >= 20).sum())}
pair = collections.Counter()
for t, p in zip(y[~hit].tolist(), order[~hit, 0].tolist()):
    pair[(min(t, p), max(t, p))] += 1
gl = [e["gloss"] for e in entries]
rep["top_confusions"] = [f"{gl[a]}~{gl[b]}:{c}" for (a, b), c in pair.most_common(40)]
rep["errors_in_pairs_seen_2plus"] = int(sum(c for c in pair.values() if c >= 2))
rep["errors_total"] = int((~hit).sum())

# 5. Train clips per class against accuracy (the tail).
n_train = collections.Counter(rows["train"][1].tolist())
n_val = collections.Counter(rows["val"][1].tolist())
per = collections.defaultdict(list)
for c, h in zip(y.tolist(), hit.tolist()):
    per[n_train[c] + n_val[c]].append(h)
grp = collections.defaultdict(list)
for k, v in per.items():
    grp["<=6" if k <= 6 else "7-9" if k <= 9 else "10-14" if k <= 14 else ">=15"] += v
rep["top1_by_wlasl_trainval_clips"] = {k: dict(n=len(v), top1=round(100 * float(np.mean(v)), 1)) for k, v in sorted(grp.items())}

pathlib.Path(sys.argv[3]).write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
