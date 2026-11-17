"""Within-set answer rule: keep the model's probability for each same-sign set, and split it between the
set's glosses by a blend of the model's own split and the WLASL clip counts.

  p'(g) = P(S(g)) * p(g)^gamma * r(g)^beta / sum_{h in S(g)} p(h)^gamma * r(h)^beta

Chosen on WLASL val with a model that never trained on val; reported on test; then applied unchanged to the
shipped model. Usage: setprior.py <repo> <sets.pt> <out.json>
"""
import collections
import json
import pathlib
import sys

import torch

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "scripts"))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import model as mv  # noqa: E402
import train as tv  # noqa: E402

N = 2000
dev = "cuda"
entries = json.load(open(repo / "data/index/WLASL_v0.3.json", encoding="utf-8"))[:N]
def graph(splits):
    """Same-sign sets from source segments listed under two glosses, using only instances of the given splits,
    so a test listing never creates a link."""
    by_url = collections.defaultdict(list)
    for c, e in enumerate(entries):
        for i in e["instances"]:
            if i["split"] in splits:
                by_url[i["url"]].append((c, i["frame_start"], i["frame_end"]))
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = set()
    for items in by_url.values():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (ca, sa, ea), (cb, sb, eb) = items[a], items[b]
                if ca == cb:
                    continue
                if ea != -1 and eb != -1:
                    inter, union = min(ea, eb) - max(sa, sb), max(ea, eb) - min(sa, sb)
                    if inter <= 0 or inter / max(union, 1) < 0.5:
                        continue
                edges.add((min(ca, cb), max(ca, cb)))
    for a, b in edges:
        parent[find(a)] = find(b)
    _, s = torch.unique(torch.tensor([find(c) for c in range(N)]), return_inverse=True)
    return s, len(edges)


comp, n_edges = graph(("train",))
GRAPH_INFO = {}


def variant(max_size):
    """Sets larger than max_size are chains of loosely related words: treat their members as singletons."""
    size = torch.bincount(comp)[comp]
    s = comp.clone()
    big = size > max_size
    s[big] = comp.max() + 1 + torch.arange(int(big.sum()))
    _, s = torch.unique(s, return_inverse=True)
    return s


def counts(trainval, manifest):
    wl, mix = torch.zeros(N), torch.zeros(N)
    for c, e in enumerate(entries):
        for i in e["instances"]:
            if i["split"] == "train":
                wl[c] += 1
                mix[c] += 2          # the clip and its degraded copy
            elif i["split"] == "val" and trainval:
                wl[c] += 1
                mix[c] += 1
    for row in json.load(open(repo / "data/aslcitizen" / manifest, encoding="utf-8")):
        if row["cls"] < N:
            mix[row["cls"]] += 1
    return wl, mix


@torch.no_grad()
def probs_for(ckpt, cache, rows, split):
    ck = torch.load(ckpt, weights_only=False)
    m = mv.build(ck["cfg"], ck["classes"]).to(dev)
    m.load_state_dict(ck["state"])
    ids, y = rows[split]
    data = {k: cache[k][ids].to(dev) for k in ("X", "W", "S", "L")}
    p = tv.predict_logits(m.eval(), data, torch.arange(len(y), device=dev), ck["cfg"], False, tv.flip_perm(ck["face_flip"]).to(dev))
    return p[:, :N].double().cpu(), y


def rule(p, sets, r, gamma, beta):
    n_sets = int(sets.max()) + 1
    mass = torch.zeros(len(p), n_sets, dtype=p.dtype).index_add_(1, sets, p)
    w = p.clamp_min(1e-12).pow(gamma) * r.double().clamp_min(0.5).pow(beta)[None]
    norm = torch.zeros(len(p), n_sets, dtype=p.dtype).index_add_(1, sets, w)
    return mass[:, sets] * w / norm[:, sets]


def acc(p, y):
    return round(100 * (p.argmax(1) == y).double().mean().item(), 2)


cache = torch.load(repo / "data/cache_hol2000w.pt", weights_only=False)
rows = tv.select(cache, entries, N)
rep = {}

tune = repo / "runs/prior_trainonly_s120/best.pt"
wl, mix = counts(False, "manifest_trainval.json")
pv, yv = probs_for(tune, cache, rows, "val")
pt, yt = probs_for(tune, cache, rows, "test")
rep["tune_model"] = dict(val=acc(pv, yv), test=acc(pt, yt))
grid = []
for cap in (2, 3, 4, 12):
    sets = variant(cap)
    for ratio, r in (("wlasl_count", wl), ("wlasl_over_mix", wl / mix.clamp_min(1))):
        for gamma in (0.0, 0.25, 0.5, 0.75, 1.0):
            for beta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                if gamma == 1.0 and beta == 0.0:
                    continue
                grid.append(dict(cap=cap, ratio=ratio, gamma=gamma, beta=beta, val=acc(rule(pv, sets, r, gamma, beta), yv),
                                 test=acc(rule(pt, sets, r, gamma, beta), yt)))
grid.sort(key=lambda g: -g["val"])
rep["top_by_val"] = grid[:12]
best = grid[0]
rep["chosen_on_val"] = best
rep["plain_gamma0_beta1_cap12"] = next(g for g in grid if g["cap"] == 12 and g["ratio"] == "wlasl_count" and g["gamma"] == 0.0 and g["beta"] == 1.0)

rep["graph_train_only"] = dict(edges=n_edges, multi_sets=int((torch.bincount(comp) > 1).sum()),
                               glosses_in_multi_sets=int((torch.bincount(comp)[comp] > 1).sum()))

# The shipped model: test only, with the rule chosen above and no further choice. It trained on train+val, so
# its sets may use train and val listings; test listings are still excluded.
wl2, mix2 = counts(True, "manifest.json")
ps, ys = probs_for(repo / "runs/soupAll8/best.pt", cache, rows, "test")
r2 = wl2 if best["ratio"] == "wlasl_count" else wl2 / mix2.clamp_min(1)
comp, n_edges = graph(("train", "val"))
size = torch.bincount(comp)[comp]
multi = size[ys] > 1
pred = ps.argmax(1)
mass = torch.zeros(len(ps), int(comp.max()) + 1, dtype=ps.dtype).index_add_(1, comp, ps)
rep["graph_trainval"] = dict(edges=n_edges, multi_sets=int((torch.bincount(comp) > 1).sum()), glosses_in_multi_sets=int((size > 1).sum()),
                             share_of_test_clips_in_multi_sets=round(100 * multi.double().mean().item(), 1),
                             shipped_strict_top1=acc(ps, ys), shipped_same_set_top1=round(100 * (comp[pred] == comp[ys]).double().mean().item(), 2),
                             shipped_set_top1_summed=round(100 * (mass.argmax(1) == comp[ys]).double().mean().item(), 2),
                             strict_on_singleton_clips=round(100 * (pred == ys)[~multi].double().mean().item(), 2),
                             strict_on_multi_clips=round(100 * (pred == ys)[multi].double().mean().item(), 2))
after = rule(ps, variant(best["cap"]), r2, best["gamma"], best["beta"])
rep["graph_trainval"]["strict_on_multi_clips_after_rule"] = round(100 * (after.argmax(1) == ys)[multi].double().mean().item(), 2)
torch.save(dict(set_of=variant(best["cap"]), wl=wl2, gamma=best["gamma"], beta=best["beta"], cap=best["cap"]), pathlib.Path(sys.argv[3]).with_suffix(".pt"))
order = after.argsort(1, descending=True)[:, :5]
rep["shipped_soupAll8_test"] = dict(before=acc(ps, ys), after=acc(after, ys),
                                    top5_before=round(100 * (ps.argsort(1, descending=True)[:, :5] == ys[:, None]).any(1).double().mean().item(), 2),
                                    top5_after=round(100 * (order == ys[:, None]).any(1).double().mean().item(), 2))
# How much of the gain is the global prior (the rejected label-shift correction) rather than the set structure?
for al in (0.5, 1.0):
    g = ps * (wl2 / mix2.clamp_min(1)).double().pow(al)[None]
    rep[f"shipped_global_prior_alpha_{al}"] = acc(g, ys)
pathlib.Path(sys.argv[3]).write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
