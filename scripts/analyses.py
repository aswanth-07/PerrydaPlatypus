"""Zero-training analyses on cached keypoints (read-only on the repo; one JSON report per analysis).

  sets    same-sign sets from shared source segments in the WLASL index; set-level accuracy of a checkpoint
  frames  accuracy against landmarked frames, fixed budgets and a margin stop rule (nested 9/17/33 frames)
  knn     exemplar memory beside the head: tuned on unseen ASL Citizen signers, reported on WLASL test
  align   unmatched ASL Citizen classes that a 2,000-word model maps onto one WLASL word

Usage: analyses.py <repo> <what> <ckpt> <out.json>
"""
import collections
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

repo = pathlib.Path(sys.argv[1])
what, ckpt_path, out_path = sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
sys.path.insert(0, str(repo / "scripts"))
import model_v2 as mv  # noqa: E402
import train_v2 as tv  # noqa: E402

dev = "cuda"
entries = json.load(open(repo / "data/index/WLASL_v0.3.json", encoding="utf-8"))[:2000]
gl = [e["gloss"] for e in entries]


def load_model(path):
    ck = torch.load(path, weights_only=False)
    m = mv.build(ck["cfg"], ck["classes"]).to(dev)
    m.load_state_dict(ck["state"])
    return m.eval(), ck["cfg"], tv.flip_perm(ck["face_flip"]).to(dev)


@torch.no_grad()
def run(model, cfg, perm, data, ids, embed=False):
    out = []
    for chunk in ids.split(512):
        X, W, S, L = (data[k][chunk].to(dev) for k in ("X", "W", "S", "L"))
        x, w, s = tv.sample_time(X, W, S, L, cfg["t_in"], False)
        if cfg["canon"]:
            x, w, s = tv.apply_flip(x, w, s, tv.dominant_left(x, s, torch.full_like(L, cfg["t_in"]), cfg.get("canon_rel", False)), perm)
        e = model.embed(x, w, s)
        out.append(F.normalize(e, dim=-1).cpu() if embed else torch.softmax(model.head(e), -1)[:, :2000].cpu())
    return torch.cat(out)


def same_sign_sets():
    """Union-find over glosses that share one source segment (same url and overlapping frame range)."""
    by_url = collections.defaultdict(list)
    for c, e in enumerate(entries):
        for i in e["instances"]:
            by_url[i["url"]].append((c, i["frame_start"], i["frame_end"], i["split"], i["video_id"]))
    parent = list(range(len(entries)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = collections.Counter()
    for url, items in by_url.items():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ca, sa, ea, _, _ = items[a]
                cb, sb, eb, _, _ = items[b]
                if ca == cb:
                    continue
                whole = ea == -1 or eb == -1
                if not whole:
                    inter = min(ea, eb) - max(sa, sb)
                    union = max(ea, eb) - min(sa, sb)
                    if inter <= 0 or inter / max(union, 1) < 0.5:
                        continue
                edges[(min(ca, cb), max(ca, cb))] += 1
    for (a, b) in edges:
        parent[find(a)] = find(b)
    comp = collections.defaultdict(list)
    for c in range(len(entries)):
        comp[find(c)].append(c)
    set_of = [0] * len(entries)
    for k, (_, members) in enumerate(sorted(comp.items())):
        for c in members:
            set_of[c] = k
    return set_of, edges, [m for m in comp.values() if len(m) > 1]


def test_rows():
    cache = torch.load(repo / "data/cache_hol2000w.pt", weights_only=False)
    rows = tv.select(cache, entries, 2000)
    return cache, rows


if what == "sets":
    set_of, edges, multi = same_sign_sets()
    cache, rows = test_rows()
    ids, y = rows["test"]
    model, cfg, perm = load_model(ckpt_path)
    probs = run(model, cfg, perm, cache, ids)
    sets = torch.tensor(set_of)
    n_sets = int(sets.max()) + 1
    pred = probs.argmax(1)
    strict = (pred == y).float().mean().item()
    lenient = (sets[pred] == sets[y]).float().mean().item()
    set_probs = torch.zeros(len(y), n_sets).index_add_(1, sets, probs)
    set_pred = set_probs.argmax(1)
    set_top1 = (set_pred == sets[y]).float().mean().item()
    in_multi = torch.tensor([len([1 for m in multi if c in m]) > 0 for c in range(2000)])
    tm = in_multi[y]
    sizes = collections.Counter(len(m) for m in multi)
    rep = dict(
        edges=len(edges), sets_with_2plus_glosses=len(multi), glosses_in_them=int(sum(len(m) for m in multi)),
        set_size_hist={str(k): v for k, v in sorted(sizes.items())}, largest_set=[gl[c] for c in max(multi, key=len)][:40],
        share_of_test_clips_in_multi_sets=round(100 * tm.float().mean().item(), 1),
        strict_top1=round(100 * strict, 2), top1_same_set_counts_as_right=round(100 * lenient, 2),
        top1_over_sets_summed_probs=round(100 * set_top1, 2),
        strict_top1_on_clips_in_multi_sets=round(100 * (pred == y)[tm].float().mean().item(), 2),
        strict_top1_on_clips_in_singleton_sets=round(100 * (pred == y)[~tm].float().mean().item(), 2),
        set_top1_on_clips_in_multi_sets=round(100 * (set_pred == sets[y])[tm].float().mean().item(), 2),
        mean_top_prob_gloss=round(probs.max(1).values.mean().item(), 3), mean_top_prob_set=round(set_probs.max(1).values.mean().item(), 3),
        examples=[" = ".join(gl[c] for c in m) for m in sorted(multi, key=len)[:25]],
        strongest_edges=[f"{gl[a]}={gl[b]} ({n})" for (a, b), n in edges.most_common(25)])
    torch.save(dict(set_of=set_of, multi=multi), out_path.with_suffix(".pt"))

elif what == "frames":
    cache, rows = test_rows()
    ids, y = rows["test"]
    model, cfg, perm = load_model(ckpt_path)
    Xf, Wf, Sf, Lf = (cache[k][ids] for k in ("X", "W", "S", "L"))

    def thin(n):
        pos = (torch.linspace(0, 1, n)[None] * (Lf[:, None] - 1).clamp_min(0).float()).round().long()
        r = torch.arange(len(Lf))[:, None]
        keep = Lf > n
        d = {}
        for k, Z in (("X", Xf), ("W", Wf), ("S", Sf)):
            z = Z.clone()
            z[keep, :n] = Z[r, pos][keep]
            d[k] = z
        d["L"] = torch.where(keep, torch.full_like(Lf, n), Lf)
        return d

    local = torch.arange(len(ids))
    rep = dict(mean_frames_in_cache=round(Lf.float().mean().item(), 1), fixed={}, margin_stop=[])
    P = {}
    for n in (6, 9, 12, 17, 24, 33, 48, 96):
        P[n] = run(model, cfg, perm, thin(n), local)
        used = torch.minimum(Lf, torch.tensor(n)).float().mean().item()
        rep["fixed"][str(n)] = dict(top1=round(100 * (P[n].argmax(1) == y).float().mean().item(), 2), mean_frames=round(used, 1))
    rounds = (9, 17, 33)
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        done = torch.zeros(len(y), dtype=torch.bool)
        pred = torch.zeros(len(y), dtype=torch.long)
        cost = torch.zeros(len(y))
        for k, n in enumerate(rounds):
            top2 = P[n].topk(2, 1).values
            stop = (top2[:, 0] - top2[:, 1] >= tau) | (k == len(rounds) - 1)
            now = stop & ~done
            pred[now] = P[n].argmax(1)[now]
            cost[now] = torch.minimum(Lf, torch.tensor(n)).float()[now]
            done |= now
        rep["margin_stop"].append(dict(tau=tau, top1=round(100 * (pred == y).float().mean().item(), 2), mean_frames=round(cost.mean().item(), 1),
                                       stopped_at_9=round(100 * (cost <= 9).float().mean().item(), 1)))

elif what == "knn":
    model, cfg, perm = load_model(ckpt_path)
    cache, rows = test_rows()
    tr = torch.cat([rows["train"][0], rows["val"][0]])
    tr_y = torch.cat([rows["train"][1], rows["val"][1]])
    store_w = run(model, cfg, perm, cache, tr, embed=True)
    te, te_y = rows["test"]
    q_w = run(model, cfg, perm, cache, te, embed=True)
    p_w = run(model, cfg, perm, cache, te)
    del cache
    ac = torch.load(repo / "data/cache_aslc2000.pt", weights_only=False)

    def aslc(manifest):
        cls = {"aslc_" + pathlib.Path(c["file"]).stem: c["cls"] for c in json.load(open(repo / "data/aslcitizen" / manifest, encoding="utf-8"))}
        r = [(n, cls[v]) for n, v in enumerate(ac["vid"]) if cls.get(v, 2000) < 2000]
        return torch.tensor([a for a, _ in r]), torch.tensor([b for _, b in r])

    a_tr, a_tr_y = aslc("manifest_trainval.json")
    a_te, a_te_y = aslc("manifest_test.json")
    store_a = run(model, cfg, perm, ac, a_tr, embed=True)
    q_a = run(model, cfg, perm, ac, a_te, embed=True)
    p_a = run(model, cfg, perm, ac, a_te)

    def knn_probs(q, store, sy, k, temp):
        out = torch.zeros(len(q), 2000)
        st, syd = store.to(dev), sy.to(dev)
        for a in range(0, len(q), 2048):
            sim, idx = (q[a:a + 2048].to(dev) @ st.T).topk(k, 1)
            wgt = torch.softmax(sim / temp, 1)
            out[a:a + 2048] = torch.zeros(len(sim), 2000, device=dev).scatter_add_(1, syd[idx], wgt).cpu()
        return out

    def acc(p, y):
        return round(100 * (p.argmax(1) == y).float().mean().item(), 2)

    rep = dict(head_only=dict(unseen=acc(p_a, a_te_y), wlasl_test=acc(p_w, te_y)), grid=[])
    stores = {"wlasl": (store_w, tr_y), "wlasl+aslc": (torch.cat([store_w, store_a]), torch.cat([tr_y, a_tr_y]))}
    for name, (st, sy) in stores.items():
        for k in (8, 32):
            for temp in (0.03, 0.07):
                ka, kw = knn_probs(q_a, st, sy, k, temp), knn_probs(q_w, st, sy, k, temp)
                for lam in (0.1, 0.2, 0.35, 0.5):
                    rep["grid"].append(dict(store=name, k=k, temp=temp, lam=lam, unseen=acc((1 - lam) * p_a + lam * ka, a_te_y),
                                            wlasl_test=acc((1 - lam) * p_w + lam * kw, te_y)))
    rep["store_rows"] = {k: len(v[1]) for k, v in stores.items()}
    best = max(rep["grid"], key=lambda r: r["unseen"])
    rep["chosen_on_unseen"] = best

elif what == "align":
    model, cfg, perm = load_model(ckpt_path)
    ac = torch.load(repo / "data/cache_aslc_all.pt", weights_only=False)
    man = json.load(open(repo / "data/aslcitizen/manifest.json", encoding="utf-8"))
    info = {"aslc_" + pathlib.Path(c["file"]).stem: c for c in man}
    r = [(n, info[v]["cls"]) for n, v in enumerate(ac["vid"]) if v in info and info[v]["cls"] >= 2000]
    ids = torch.tensor([a for a, _ in r])
    cls = np.array([b for _, b in r])
    names = {c["cls"]: c["gloss"] for c in man}
    probs = run(model, cfg, perm, ac, ids)
    pred, conf = probs.argmax(1).numpy(), probs.max(1).values.numpy()
    table = []
    for c in sorted(set(cls.tolist())):
        m = cls == c
        top, n = collections.Counter(pred[m].tolist()).most_common(1)[0]
        table.append(dict(aslc=names[c], wlasl=gl[top], clips=int(m.sum()), agree=round(n / m.sum(), 2), conf=round(float(conf[m & (pred == top)].mean()), 2)))
    rep = dict(unmatched_classes=len(table), clips=len(ids))
    for thr in (0.5, 0.6, 0.7, 0.8):
        hit = [t for t in table if t["agree"] >= thr]
        rep[f"agree>={thr}"] = dict(classes=len(hit), clips=int(sum(t["clips"] for t in hit)), wlasl_words=len({t["wlasl"] for t in hit}))
    rep["examples_high"] = [f"{t['aslc']}->{t['wlasl']} {t['agree']}" for t in sorted(table, key=lambda t: -t["agree"])[:60]]
    rep["examples_mid"] = [f"{t['aslc']}->{t['wlasl']} {t['agree']}" for t in sorted(table, key=lambda t: abs(t["agree"] - 0.6))[:30]]
    out_path.with_suffix(".table.json").write_text(json.dumps(table), encoding="utf-8")

out_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
