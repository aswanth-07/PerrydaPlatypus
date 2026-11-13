"""Score checkpoints (an ensemble if several) the same way every time, so runs can be compared line by line.

  wlasl     the 2,878-clip WLASL2000 test split: strict top-1/top-5, with the same-sign answer rule, and counting
            a gloss of the same same-sign set as right
  unseen    ASL Citizen's 11 test signers (in no training data of a measurement run), the same three scores
  prefix    top-1 when only the first 50-90% of each clip is shown (how early a sign can be called)
  robust    clean against each degraded condition on the robustness sets, strict and with the rule

Usage: eval_suite.py <ckpt.pt> [<ckpt.pt> ...] [--out f.json] [--skip unseen,prefix,robust] [--robust DIR ...]
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eval_v2  # noqa: E402
import model_v2 as mv  # noqa: E402
import signsets  # noqa: E402
import train_v2 as tv  # noqa: E402

N = 2000
ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(ckpts):
    out = []
    for c in ckpts:
        ck = torch.load(c, weights_only=False)
        m = mv.build(ck["cfg"], ck["classes"]).cuda()
        m.load_state_dict(ck["state"])
        out.append((m.eval(), ck["cfg"], tv.flip_perm(ck.get("face_flip", torch.arange(62))).cuda()))
    return out


def probs(models, data, ids, upto=1.0):
    return sum(tv.predict_logits(m, data, ids, cfg, False, perm, upto=upto)[:, :N] for m, cfg, perm in models) / len(models)


def three(p, y, set_of, counts):
    """Strict, with the answer rule, and with a same-set gloss counted as right."""
    ruled = signsets.answer_rule(p.double(), set_of, counts)
    top5 = ruled.topk(5, 1).indices
    return dict(n=len(y), top1=round(100 * (p.argmax(1) == y).float().mean().item(), 2),
                top5=round(100 * (p.topk(5, 1).indices == y[:, None]).any(1).float().mean().item(), 2),
                rule_top1=round(100 * (ruled.argmax(1) == y).float().mean().item(), 2),
                rule_top5=round(100 * (top5 == y[:, None]).any(1).float().mean().item(), 2),
                same_set_top1=round(100 * (set_of[p.argmax(1)] == set_of[y]).float().mean().item(), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--skip", default="")
    ap.add_argument("--robust", nargs="*", default=["data/kp_rob100_mod", "data/kp_rob100"])
    ap.add_argument("--robust-subset", type=int, default=N, help="classes whose test clips the robustness sets cover")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    entries = json.load(open(ROOT / "data/index/WLASL_v0.3.json", encoding="utf-8"))
    set_of, links = signsets.same_sign_sets(entries, N)
    set_of, counts = set_of.cuda(), signsets.wlasl_counts(entries, N).cuda()
    models = load(a.ckpts)
    res = dict(ckpts=a.ckpts, same_sign_links=links)

    cache = torch.load(ROOT / "data/cache_hol2000w.pt", weights_only=False)
    ids, y = tv.select(cache, entries, N)["test"]
    data = {k: cache[k][ids].cuda() for k in ("X", "W", "S", "L")}
    del cache
    local, y = torch.arange(len(y)).cuda(), y.cuda()
    res["wlasl"] = three(probs(models, data, local), y, set_of, counts)
    if "prefix" not in skip:
        res["prefix_wlasl"] = {str(u): round(100 * (probs(models, data, local, u).argmax(1) == y).float().mean().item(), 2)
                               for u in (0.5, 0.6, 0.7, 0.8, 0.9)}
    del data

    if "unseen" not in skip:
        ac = torch.load(ROOT / "data/cache_aslc2000.pt", weights_only=False)
        cls = {"aslc_" + pathlib.Path(c["file"]).stem: c["cls"]
               for c in json.load(open(ROOT / "data/aslcitizen/manifest_test.json", encoding="utf-8"))}
        rows = [(n, cls[v]) for n, v in enumerate(ac["vid"]) if cls.get(v, N) < N]
        ai = torch.tensor([r[0] for r in rows])
        ay = torch.tensor([r[1] for r in rows]).cuda()
        adata = {k: ac[k][ai].cuda() for k in ("X", "W", "S", "L")}
        del ac
        al = torch.arange(len(ay)).cuda()
        res["unseen"] = three(probs(models, adata, al), ay, set_of, counts)
        if "prefix" not in skip:
            res["prefix_unseen"] = {str(u): round(100 * (probs(models, adata, al, u).argmax(1) == ay).float().mean().item(), 2)
                                    for u in (0.5, 0.6, 0.7, 0.8, 0.9)}
        del adata

    if "robust" not in skip:
        clean_dir = ROOT / "data/kp_hol"
        for rd in a.robust:
            rd = ROOT / rd
            conds = sorted(p.name for p in rd.iterdir() if p.is_dir()) if rd.exists() else []
            vids, labels = [], []
            for c, e in enumerate(entries[:a.robust_subset]):
                for i in e["instances"]:
                    v = i["video_id"]
                    if i["split"] == "test" and (clean_dir / f"{v}.npz").exists() and all((rd / k / f"{v}.npz").exists() for k in conds):
                        vids.append(v)
                        labels.append(c)
            if not conds or not vids:
                continue
            ry, rl = torch.tensor(labels).cuda(), torch.arange(len(vids)).cuda()
            block = {}
            for cond, d in [("clean", clean_dir)] + [(k, rd / k) for k in conds]:
                p = probs(models, eval_v2.load_dir(d, vids), rl)
                ruled = signsets.answer_rule(p.double(), set_of, counts)
                block[cond] = dict(top1=round(100 * (p.argmax(1) == ry).float().mean().item(), 2),
                                   rule_top1=round(100 * (ruled.argmax(1) == ry).float().mean().item(), 2))
            for cond in block:
                block[cond]["drop"] = round(block["clean"]["top1"] - block[cond]["top1"], 2)
                block[cond]["rule_drop"] = round(block["clean"]["rule_top1"] - block[cond]["rule_top1"], 2)
            res["robust_" + rd.name] = dict(clips=len(vids), **block)

    text = json.dumps(res, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(text, encoding="utf-8")
    brief = {k: v for k, v in res.items() if k in ("wlasl", "unseen", "prefix_wlasl")}
    for k, v in res.items():
        if k.startswith("robust_"):
            brief[k] = {c: (r["top1"], r["drop"]) if isinstance(r, dict) else r for c, r in v.items()}
    print(json.dumps(brief))


if __name__ == "__main__":
    main()
