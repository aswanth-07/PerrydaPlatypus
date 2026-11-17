"""How well can a user teach the model a new sign from a few examples? (the teach-a-sign feature, offline)

The exported model gives a pooled embedding per clip. A taught sign is the mean of the unit-length embeddings of
its k examples (a prototype); a clip is called that sign when its cosine to the prototype is the largest among
the taught signs and at least a threshold. This measures, over many random episodes:

  accuracy    N-way k-shot: the right taught sign among N, for clips by signers who gave none of the examples
  recall      the same, but the clip must also pass the threshold
  false       how often a clip that is NOT a taught sign passes the threshold: clips of known vocabulary words
              (WLASL test split) and clips of other signs nobody taught
  The threshold is one number for all episodes, set where known-word clips pass at 1%, 2% and 5%.

New signs are ASL Citizen signs outside the 2,000-word vocabulary (cls >= 2000 in the manifest). A checkpoint
whose lineage trained on those signs as union classes has seen them, so its numbers are optimistic; a model
trained on WLASL only has never seen them.

--cache/--manifest take the new signs from another set landmarked the same way, such as INCLUDE (Indian Sign
Language, scripts/include_prep.py); its manifest's signer column decides which clips may be queries.

Usage: fewshot_eval.py CKPT [CKPT ...] [--ways 20] [--shots 1 3 5] [--episodes 300] [--out f.json]
       [--cache data/cache_include.pt --manifest data/include/manifest.json]
"""
import argparse
import json
import pathlib
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import model as mv  # noqa: E402
import train as tv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


@torch.no_grad()
def embed(models, cache, ids):
    out = []
    for chunk in ids.split(512):
        X, W, S, L = (cache[k][chunk].cuda() for k in ("X", "W", "S", "L"))
        e = 0
        for m, cfg, perm in models:
            x, w, s = tv.sample_time(X, W, S, L, cfg["t_in"], False, crop=0.0, power=1.0)
            if cfg["canon"]:
                left = tv.dominant_left(x, s, torch.full_like(L, cfg["t_in"]), cfg.get("canon_rel", False))
                x, w, s = tv.apply_flip(x, w, s, left, perm)
            e = e + m.embed(x, w, s)
        out.append((e / len(models)).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--ways", type=int, default=20)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--cache", default="data/cache_aslc_all.pt", help="keypoint cache holding the new signs")
    ap.add_argument("--manifest", default="data/aslcitizen/manifest.json", help="rows with file, cls (>= 2000 = new), signer")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    models = []
    for c in a.ckpts:
        ck = torch.load(c, weights_only=False)
        m = mv.build(ck["cfg"], ck["classes"]).cuda()
        m.load_state_dict(ck["state"])
        models.append((m.eval(), ck["cfg"], tv.flip_perm(ck.get("face_flip", torch.arange(62))).cuda()))

    # New signs: ASL Citizen clips outside the vocabulary, with the signer of each.
    ac = torch.load(ROOT / a.cache, weights_only=False)
    man = {"aslc_" + pathlib.Path(c["file"]).stem: c for c in json.load(open(ROOT / a.manifest, encoding="utf-8"))}
    rows = [(n, man[v]["cls"], man[v]["signer"]) for n, v in enumerate(ac["vid"]) if v in man and man[v]["cls"] >= 2000]
    new = embed(models, ac, torch.tensor([r[0] for r in rows]))
    del ac
    by_cls = {}
    for i, (_, c, s) in enumerate(rows):
        by_cls.setdefault(c, []).append((i, s))

    # Known vocabulary: WLASL train clips give the centre, test clips are the distractors.
    wc = torch.load(ROOT / "data/cache_hol2000w.pt", weights_only=False)
    sel = tv.select(wc, json.load(open(ROOT / "data/index/WLASL_v0.3.json", encoding="utf-8")), 2000, None)
    centre = embed(models, wc, sel["train"][0]).mean(0)
    known = embed(models, wc, sel["test"][0])
    del wc

    if len(by_cls) < 2 * a.ways:   # a small set (INCLUDE): half the signs taught, the other half as untaught distractors
        a.ways = max(2, len(by_cls) // 2)
    rep = dict(ckpts=a.ckpts, cache=a.cache, ways=a.ways, episodes=a.episodes, new_signs=len(by_cls), new_clips=len(rows), known_clips=len(known))
    for centred in (False, True):
        unit = (lambda e: F.normalize(e - centre, dim=-1)) if centred else (lambda e: F.normalize(e, dim=-1))
        new_u, known_u = unit(new), unit(known)
        for k in a.shots:
            pool = [c for c, v in by_cls.items() if len(v) >= k + 5]
            hit, best, known_max, other_max = [], [], [], []
            for _ in range(a.episodes):
                classes = rng.sample(pool, a.ways)
                protos, queries = [], []
                for j, c in enumerate(classes):
                    clips = by_cls[c][:]
                    rng.shuffle(clips)
                    support = clips[:k]
                    signers = {s for _, s in support}
                    protos.append(F.normalize(new_u[[i for i, _ in support]].mean(0), dim=0))
                    queries += [(i, j) for i, s in clips[k:] if s not in signers]
                protos = torch.stack(protos)
                sims = new_u[[i for i, _ in queries]] @ protos.T
                top, arg = sims.max(1)
                hit.append(arg == torch.tensor([j for _, j in queries]))
                best.append(top)
                known_max.append((known_u @ protos.T).max(1).values)
                rest = [c for c in pool if c not in classes]
                others = [i for c in rng.sample(rest, min(40, len(rest))) for i, _ in by_cls[c][:5]]
                other_max.append((new_u[others] @ protos.T).max(1).values)
            hit, best, known_max, other_max = map(torch.cat, (hit, best, known_max, other_max))
            r = dict(accuracy=round(100 * hit.float().mean().item(), 2), queries=len(hit))
            for fpr in (0.01, 0.02, 0.05):
                thr = torch.quantile(known_max, 1 - fpr).item()
                r[f"known_false_{int(100 * fpr)}pct"] = dict(
                    threshold=round(thr, 3),
                    recall=round(100 * (hit & (best >= thr)).float().mean().item(), 2),
                    other_new_false=round(100 * (other_max >= thr).float().mean().item(), 2))
            rep[f"{'centred' if centred else 'plain'}_{k}shot"] = r
    print(json.dumps(rep, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
