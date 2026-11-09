"""Train SignNet on a WLASL subset on the GPU and report held-out test accuracy.

Everything after loading the cache runs on the GPU: temporal sampling, augmentation, features, model.
The checkpoint (an EMA of the weights) is chosen on validation top-1; test is read only at the end.

Usage: train_v2.py <index.json> <cache.pt> <out_dir> --subset 100 [--epochs 300] [--dim 192] [--seed 7] ...
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_v2 as dv
import model_v2 as mv

POSE_WRIST = (dv.POSE_KEEP.index(15), dv.POSE_KEEP.index(16))
FINGERS = [list(range(1 + 4 * f, 5 + 4 * f)) for f in range(5)]


def select(cache, entries, subset, train_ids=None):
    """Rows per split. Degraded copies (<vid>__a<k>) join the training split only, never val or test."""
    pos, copies = {}, {}
    for n, v in enumerate(cache["vid"]):
        if "__a" in v:
            copies.setdefault(v.split("__a")[0], []).append(n)
        else:
            pos[v] = n
    rows = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for cls, e in enumerate(entries[:subset]):
        for i in e["instances"]:
            v = i["video_id"]
            if v not in pos or (i["split"] == "train" and train_ids is not None and v not in train_ids):
                continue
            extra = copies.get(v, []) if i["split"] == "train" else []
            for r in [pos[v]] + extra:
                rows[i["split"]][0].append(r)
                rows[i["split"]][1].append(cls)
    return {k: (torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)) for k, (a, b) in rows.items()}


def shrink(cache, t_max):
    """Thin every clip longer than t_max to t_max evenly spaced frames and cut the padding: about half the memory
    at t_max 48, with no effect on inference (the model resamples to 64 steps; predict() landmarks 24 frames)."""
    L = cache["L"]
    if cache["X"].shape[1] <= t_max:
        return cache
    pos = (torch.linspace(0, 1, t_max)[None] * (L[:, None] - 1).clamp_min(0).float()).round().long()
    idx = torch.where((L > t_max)[:, None], pos, torch.arange(t_max)[None].expand(len(L), -1))
    r = torch.arange(len(L))[:, None]
    for k in ("X", "W", "S"):
        cache[k] = cache[k][r, idx].contiguous()
    cache["L"] = L.clamp(max=t_max)
    return cache


def merge(cache, extra):
    for k in ("X", "W", "S", "L"):
        cache[k] = torch.cat([cache[k], extra[k]])
    cache["vid"] = list(cache["vid"]) + list(extra["vid"])
    return cache


def flip_perm(face_flip):
    perm = list(range(dv.P))
    for a, b in dv.POSE_FLIP:
        perm[a], perm[b] = b, a
    for k in range(21):
        perm[dv.LH[0] + k], perm[dv.RH[0] + k] = dv.RH[0] + k, dv.LH[0] + k
    for k, j in enumerate(face_flip.tolist()):
        perm[dv.FACE[0] + k] = dv.FACE[0] + j
    return torch.tensor(perm)


def apply_flip(x, w, s, mask, perm):
    if not bool(mask.any()):
        return x, w, s
    xf = x[:, :, perm].clone()
    xf[..., 0] = -xf[..., 0]
    wf = w[:, :, [1, 0]].clone()
    wf[..., 0] = -wf[..., 0]
    sf = s[..., [0, 2, 1, 3]]
    m4 = mask[:, None, None, None]
    return torch.where(m4, xf, x), torch.where(mask[:, None, None, None, None], wf, w), torch.where(mask[:, None, None], sf, s)


def dominant_left(x, s, L, relative=False):
    """True where the person's left hand does more of the signing: more frames seen, longer wrist path.
    relative=True measures the wrist path against the shoulder midpoint, so camera shake moves neither hand."""
    t = x.shape[1]
    valid = (torch.arange(t, device=x.device)[None] < L[:, None]).float()
    mid = (x[:, :, dv.SHOULDERS[0], :2] + x[:, :, dv.SHOULDERS[1], :2]) / 2 if relative else 0
    score = []
    for slot, wi in ((1, POSE_WRIST[0]), (2, POSE_WRIST[1])):
        wr = x[:, :, wi, :2] - mid
        path = ((wr[:, 1:] - wr[:, :-1]).norm(dim=-1) * valid[:, 1:]).sum(1)
        score.append((s[..., slot] * valid).sum(1) / L.clamp_min(1) + 2.0 * path)
    return score[0] > score[1]


def sample_time(X, W, S, L, t_in, train, crop=0.0, power=1.0):
    """Resample each clip to t_in steps. Training jitters the window and the speed per clip."""
    b = len(L)
    dev = X.device
    grid = torch.linspace(0, 1, t_in, device=dev).unsqueeze(0).expand(b, -1)
    if train:
        start = torch.rand(b, 1, device=dev) * 0.15
        end = 1 - torch.rand(b, 1, device=dev) * 0.15
        pw = torch.empty(b, 1, device=dev).uniform_(0.7, 1.4)
        grid = start + (end - start) * grid.pow(pw)
    else:
        grid = crop + (1 - 2 * crop) * grid.pow(power)
    pos = grid * (L[:, None].float() - 1).clamp_min(0)
    lo = pos.floor().long()
    hi = torch.minimum(lo + 1, (L[:, None] - 1).clamp_min(0))
    a = (pos - lo.float())
    r = torch.arange(b, device=dev)[:, None]

    def lerp(Z):
        z0, z1 = Z[r, lo].float(), Z[r, hi].float()
        aa = a.view(b, t_in, *([1] * (Z.dim() - 2)))
        return z0 + (z1 - z0) * aa

    return lerp(X), lerp(W), lerp(S)


def sparsify(X, W, S, L, lo=16, hi=32):
    """Keep only n evenly spaced frames per clip (n drawn per batch), as predict() landmarks just 24 frames."""
    n = int(torch.randint(lo, hi + 1, (1,)))
    b = len(L)
    dev = L.device
    jitter = (torch.rand(b, n, device=dev) - 0.5) * 0.8
    pos = (torch.linspace(0, 1, n, device=dev)[None] * (L[:, None] - 1).clamp_min(0).float() + jitter).round()
    idx = torch.minimum(pos.clamp_min(0).long(), (L[:, None] - 1).clamp_min(0))
    r = torch.arange(b, device=dev)[:, None]
    keep = L > n
    Lx = torch.where(keep, torch.full_like(L, n), L)
    pad = lambda Z: torch.where(keep.view(-1, *([1] * (Z.dim() - 1))), torch.nn.functional.pad(Z[r, idx], (0, 0) * (Z.dim() - 2) + (0, Z.shape[1] - n)), Z)
    return pad(X), pad(W), pad(S), Lx


def span_mask(b, t, p, dev, lo=4, hi_frac=0.5):
    hit = torch.rand(b, device=dev) < p
    n = torch.randint(lo, max(lo + 1, int(t * hi_frac)), (b,), device=dev)
    a = (torch.rand(b, device=dev) * (t - n + 1).float()).long()
    ar = torch.arange(t, device=dev)[None]
    return hit[:, None] & (ar >= a[:, None]) & (ar < (a + n)[:, None])  # B x T


def augment(x, w, s, perm, a):
    b, t = x.shape[:2]
    dev = x.device
    x, w, s = apply_flip(x, w, s, torch.rand(b, device=dev) < a.flip, perm)
    present = (x.abs().sum(-1, keepdim=True) > 0).float()  # never-seen groups stay at zero

    # Camera: roll, scale, x squeeze (yaw-like), shear, shift.
    th = torch.empty(b, device=dev).uniform_(-a.rot, a.rot)
    sc = torch.empty(b, device=dev).uniform_(1 - a.scale, 1 + a.scale)
    sx = torch.empty(b, device=dev).uniform_(0.75, 1.1)
    sh = torch.empty(b, device=dev).uniform_(-a.shear, a.shear)
    c, sn = th.cos(), th.sin()
    m00, m01 = c * sc * sx, c * sh * sc - sn * sc
    m10, m11 = sn * sc * sx, sn * sh * sc + c * sc
    px, py = x[..., 0], x[..., 1]
    nx = m00[:, None, None] * px + m01[:, None, None] * py
    ny = m10[:, None, None] * px + m11[:, None, None] * py
    shift = torch.empty(b, 1, 1, 2, device=dev).uniform_(-a.shift, a.shift)
    x = torch.stack([nx, ny, x[..., 2] * sc[:, None, None]], -1)
    x[..., :2] = x[..., :2] + shift
    # Moving head: a smooth drift of the whole view that changes over the clip.
    if a.shake > 0:
        tt = torch.linspace(0, 1, t, device=dev)[None, :]
        for c in (0, 1):
            amp = torch.empty(b, 1, device=dev).uniform_(0, a.shake)
            freq = torch.empty(b, 1, device=dev).uniform_(0.5, 2.5)
            phase = torch.empty(b, 1, device=dev).uniform_(0, 2 * np.pi)
            x[..., c] = x[..., c] + (amp * torch.sin(2 * np.pi * freq * tt + phase))[:, :, None]

    # Handshape: small rotation and scale of each hand about its wrist.
    for lo, hi in (dv.LH, dv.RH):
        h = x[:, :, lo:hi]
        wr = h[:, :, :1]
        rel = h - wr
        ph = torch.empty(b, device=dev).uniform_(-a.hand_rot, a.hand_rot)[:, None, None]
        hs = torch.empty(b, device=dev).uniform_(0.9, 1.1)[:, None, None]
        rx = (ph.cos() * rel[..., 0] - ph.sin() * rel[..., 1]) * hs
        ry = (ph.sin() * rel[..., 0] + ph.cos() * rel[..., 1]) * hs
        x[:, :, lo:hi] = wr + torch.stack([rx, ry, rel[..., 2] * hs], -1)
    x = (x + torch.randn_like(x) * a.noise) * present

    # Occlusion: a hand lost for a stretch, a finger lost for the clip, the face lost for the clip.
    for slot, (lo, hi) in ((1, dv.LH), (2, dv.RH)):
        m = span_mask(b, t, a.hand_drop, dev)
        keep = (~m).float()
        x[:, :, lo:hi] = x[:, :, lo:hi] * keep[:, :, None, None]
        w[:, :, slot - 1] = w[:, :, slot - 1] * keep[:, :, None, None]
        s[..., slot] = s[..., slot] * keep
    fd = torch.rand(b, device=dev) < a.finger_drop
    if bool(fd.any()):
        hand = torch.randint(0, 2, (b,), device=dev)
        finger = torch.randint(0, 5, (b,), device=dev)
        pmask = torch.zeros(b, dv.P, device=dev)
        wmask = torch.zeros(b, 2, 21, device=dev)
        for f in range(5):
            for k in FINGERS[f]:
                sel = fd & (finger == f)
                pmask[:, dv.LH[0] + k] += (sel & (hand == 0)).float()
                pmask[:, dv.RH[0] + k] += (sel & (hand == 1)).float()
                wmask[:, 0, k] += (sel & (hand == 0)).float()
                wmask[:, 1, k] += (sel & (hand == 1)).float()
        x = x * (1 - pmask.clamp(max=1))[:, None, :, None]
        w = w * (1 - wmask.clamp(max=1))[:, None, :, :, None]
    face_off = (torch.rand(b, device=dev) < a.face_drop).float()
    x[:, :, dv.FACE[0]:dv.FACE[1]] *= (1 - face_off)[:, None, None, None]
    s[..., 3] = s[..., 3] * (1 - face_off)[:, None]
    pd = (torch.rand(b, 1, dv.P, 1, device=dev) < a.point_drop).float()
    x = x * (1 - pd)
    return x.clamp(-dv.CLAMP, dv.CLAMP), w.clamp(-dv.CLAMP, dv.CLAMP), s


@torch.no_grad()
def predict_logits(model, data, ids, cfg, tta, perm, keep_none=False):
    """Class probabilities; a model trained with a "no sign" class drops that column unless keep_none."""
    probs = _predict(model, data, ids, cfg, tta, perm)
    return probs if keep_none or not cfg.get("none_class") else probs[:, :-1]


def _predict(model, data, ids, cfg, tta, perm):
    model.eval()
    out = []
    for chunk in ids.split(512):
        X, W, S, L = (data[k][chunk] for k in ("X", "W", "S", "L"))
        views = [dict(crop=0.0, power=1.0)]
        if tta:
            views += [dict(crop=0.06, power=1.0), dict(crop=0.0, power=0.85), dict(crop=0.0, power=1.15)]
        probs = 0
        for v in views:
            x, w, s = sample_time(X, W, S, L, cfg["t_in"], False, **v)
            if cfg["canon"]:
                x, w, s = apply_flip(x, w, s, dominant_left(x, s, torch.full_like(L, cfg["t_in"]), cfg.get("canon_rel", False)), perm)
            probs = probs + torch.softmax(model(x, w, s), -1)
            if tta and cfg["flip"] > 0:
                xf, wf, sf = apply_flip(x, w, s, torch.ones(len(L), dtype=torch.bool, device=x.device), perm)
                probs = probs + torch.softmax(model(xf, wf, sf), -1)
        out.append(probs)
    return torch.cat(out)


def score(probs, y):
    order = probs.topk(min(5, probs.shape[1]), 1).indices
    hit1 = (order[:, 0] == y)
    per_class = {}
    for c, h in zip(y.tolist(), hit1.tolist()):
        per_class.setdefault(c, []).append(h)
    return dict(n=len(y), top1=round(100 * hit1.float().mean().item(), 2),
                top5=round(100 * (order == y[:, None]).any(1).float().mean().item(), 2),
                class_avg=round(100 * float(np.mean([np.mean(v) for v in per_class.values()])), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--train-ids")
    ap.add_argument("--aug-cache", action="append", default=[], help="cache of degraded training copies; repeatable")
    ap.add_argument("--extra", action="append", default=[], help="CACHE::MANIFEST of extra labelled clips (training only); repeatable")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--kernel", type=int, default=17)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--head-drop", type=float, default=0.4)
    ap.add_argument("--late-drop", type=float, default=0.0, help="head dropout switched on at --late-epoch")
    ap.add_argument("--late-epoch", type=int, default=0)
    ap.add_argument("--t-in", type=int, default=64)
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-world", action="store_true")
    ap.add_argument("--no-z", action="store_true")
    ap.add_argument("--canon", action="store_true", help="flip every clip so the more active hand is the right one")
    ap.add_argument("--canon-rel", action="store_true", help="with --canon: measure wrist paths against the shoulders")
    ap.add_argument("--flip", type=float, default=0.5)
    ap.add_argument("--rot", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--shear", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=0.25)
    ap.add_argument("--hand-rot", type=float, default=0.2)
    ap.add_argument("--shake", type=float, default=0.15, help="peak moving-head drift in shoulder widths")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--hand-drop", type=float, default=0.2)
    ap.add_argument("--finger-drop", type=float, default=0.2)
    ap.add_argument("--face-drop", type=float, default=0.15)
    ap.add_argument("--point-drop", type=float, default=0.03)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.0, help="probability of mixing a batch (time-aligned, labels mixed)")
    ap.add_argument("--sparse", type=float, default=0.0, help="probability a batch is thinned to 16-32 frames per clip")
    ap.add_argument("--t-max", type=int, default=0, help="thin every cache to at most this many frames per clip")
    ap.add_argument("--none-prob", type=float, default=0.0, help="share of each batch turned into 'no sign' boundary windows")
    ap.add_argument("--head", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--init", default="", help="checkpoint to warm-start from (shape-matched tensors only)")
    ap.add_argument("--trainval", action="store_true", help="train on train+val and keep the last EMA (no selection)")
    ap.add_argument("--teachers", default="", help="comma-separated checkpoints whose averaged softmax the model also learns")
    ap.add_argument("--distill-alpha", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--phonology", default="", help="ASL-LEX signdata.csv: adds auxiliary phoneme heads (training only)")
    ap.add_argument("--phon-weight", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=5)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    entries = json.load(open(a.index, encoding="utf-8"))
    fit = (lambda c: shrink(c, a.t_max)) if a.t_max else (lambda c: c)
    cache = fit(torch.load(a.cache, weights_only=False))
    for extra in a.aug_cache:
        cache = merge(cache, fit(torch.load(extra, weights_only=False)))
    extra_cls = {}
    for spec in a.extra:  # CACHE::MANIFEST from another dataset (ASL Citizen): training rows only
        path, manifest = spec.split("::")
        cache = merge(cache, fit(torch.load(path, weights_only=False)))
        for c in json.load(open(manifest, encoding="utf-8")):
            extra_cls["aslc_" + pathlib.Path(c["file"]).stem] = c["cls"]
    train_ids = set(pathlib.Path(a.train_ids).read_text().split()) if a.train_ids else None
    rows = select(cache, entries, a.subset, train_ids)
    if extra_cls:
        add = [(n, extra_cls[v]) for n, v in enumerate(cache["vid"]) if extra_cls.get(v, a.subset) < a.subset]
        if add:
            ids_x, y_x = map(torch.tensor, zip(*add))
            rows["train"] = (torch.cat([rows["train"][0], ids_x]), torch.cat([rows["train"][1], y_x]))
        print(f"extra labelled clips in training: {len(add)}", flush=True)
    used = torch.unique(torch.cat([r[0] for r in rows.values()]))
    remap = torch.full((len(cache["vid"]),), -1, dtype=torch.long)
    remap[used] = torch.arange(len(used))
    rows = {k: (remap[i].to(dev), y.to(dev)) for k, (i, y) in rows.items()}
    data = {k: cache[k][used].to(dev) for k in ("X", "W", "S")}
    data["L"] = cache["L"][used].to(dev)
    perm = flip_perm(cache["face_flip"]).to(dev)
    print({k: len(v[0]) for k, v in rows.items()}, "format", cache["format"], flush=True)
    cache = dict(face_flip=cache["face_flip"], format=cache["format"])  # free the CPU copy; the subset lives on the GPU
    import gc
    gc.collect()

    cfg = dict(dim=a.dim, repeats=a.repeats, convs=a.convs, kernel=a.kernel, heads=a.heads, drop=a.drop,
               head_drop=a.head_drop, face=not a.no_face, world=not a.no_world, z=not a.no_z, t_in=a.t_in,
               canon=a.canon, canon_rel=a.canon_rel, flip=0.0 if a.canon else a.flip, format=cache["format"], head=a.head)
    if a.trainval:  # the deployment fit: validation clips join training, the last EMA is kept
        rows["train"] = (torch.cat([rows["train"][0], rows["val"][0]]), torch.cat([rows["train"][1], rows["val"][1]]))
    teachers = []
    for path in filter(None, a.teachers.split(",")):
        ck = torch.load(path, weights_only=False)
        t = mv.build(ck["cfg"], ck["classes"]).to(dev)
        t.load_state_dict(ck["state"])
        teachers.append(t.eval())
    n_cls = a.subset + (1 if a.none_prob else 0)   # the extra last class is "no sign" (a boundary between signs)
    cfg["none_class"] = bool(a.none_prob)
    model = mv.build(cfg, n_cls).to(dev)
    if a.init:  # warm start: every weight whose shape matches (all but the class head when vocabularies differ)
        src = torch.load(a.init, weights_only=False)["state"]
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        own.update(keep)
        model.load_state_dict(own)
        print(f"warm start from {a.init}: {len(keep)}/{len(own)} tensors", flush=True)
    if a.late_epoch:
        model.head_drop.p = 0.0
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters())
    # Auxiliary phoneme heads (Kezar et al. 2023): trained beside the sign head, never exported.
    aux, phon = None, None
    if a.phonology:
        import phonology as ph
        labels_by_type, sizes, matched = ph.load(a.phonology, [e["gloss"] for e in entries[:a.subset]])
        # Classes beyond the WLASL index (a union vocabulary from another dataset) have no phoneme labels.
        phon = {t: torch.tensor(v + [-1] * (n_cls - len(v)), device=dev) for t, v in labels_by_type.items()}
        aux = torch.nn.ModuleDict({t: torch.nn.Linear(2 * a.dim, n) for t, n in sizes.items()}).to(dev)
        print(f"phonology heads: {matched}/{a.subset} glosses labelled, {len(sizes)} phoneme types", flush=True)
    opt = torch.optim.AdamW(list(model.parameters()) + (list(aux.parameters()) if aux else []), lr=a.lr, weight_decay=a.wd)
    ids, labels = rows["train"]
    steps = max(len(ids) // a.batch, 1)
    total_steps = a.epochs * steps
    warm = a.warmup * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(k / total_steps, 1.0))))
    a.flip = cfg["flip"]
    best, best_epoch, t0 = -1.0, 0, time.time()
    for epoch in range(1, a.epochs + 1):
        if a.late_epoch and epoch == a.late_epoch:
            model.head_drop.p = a.late_drop
        model.train()
        order = torch.randperm(len(ids), device=dev)
        total = 0.0
        for k in range(steps):
            pick = order[k * a.batch:(k + 1) * a.batch]
            c = ids[pick]
            Xb, Wb, Sb, Lb = data["X"][c], data["W"][c], data["S"][c], data["L"][c]
            if a.sparse and np.random.rand() < a.sparse:
                Xb, Wb, Sb, Lb = sparsify(Xb, Wb, Sb, Lb)
            x, w, s = sample_time(Xb, Wb, Sb, Lb, a.t_in, True)
            if a.canon:
                x, w, s = apply_flip(x, w, s, dominant_left(x, s, torch.full_like(c, a.t_in), a.canon_rel), perm)
            x, w, s = augment(x, w, s, perm, a)
            y = labels[pick]
            if a.none_prob:  # boundary windows: the end of one sign spliced to the start of another -> "no sign"
                k = max(int(len(pick) * a.none_prob), 1)
                j, j2 = torch.randperm(len(pick), device=dev)[:k], torch.randperm(len(pick), device=dev)[:k]
                cut = int(torch.randint(a.t_in // 3, 2 * a.t_in // 3 + 1, (1,)))
                x, w, s = x.clone(), w.clone(), s.clone()
                x[j] = torch.cat([x[j][:, cut:], x[j2][:, :cut]], 1)
                w[j] = torch.cat([w[j][:, cut:], w[j2][:, :cut]], 1)
                s[j] = torch.cat([s[j][:, cut:], s[j2][:, :cut]], 1)
                y = y.clone()
                y[j] = a.subset
            if a.mixup and np.random.rand() < a.mixup:
                lam = float(np.random.beta(0.4, 0.4))
                j = torch.randperm(len(pick), device=dev)
                logits = model(lam * x + (1 - lam) * x[j], lam * w + (1 - lam) * w[j], lam * s + (1 - lam) * s[j])
                loss = lam * F.cross_entropy(logits, y, label_smoothing=a.smooth) + (1 - lam) * F.cross_entropy(logits, y[j], label_smoothing=a.smooth)
            else:
                emb = model.embed(x, w, s)
                logits = model.head(model.head_drop(emb))
                loss = F.cross_entropy(logits, y, label_smoothing=a.smooth)
                if teachers:
                    with torch.no_grad():
                        soft = sum(torch.softmax(t(x, w, s) / a.distill_temp, -1) for t in teachers) / len(teachers)
                    kd = F.kl_div(F.log_softmax(logits / a.distill_temp, -1), soft, reduction="batchmean") * a.distill_temp ** 2
                    loss = (1 - a.distill_alpha) * loss + a.distill_alpha * kd
                if aux is not None:
                    terms = [F.cross_entropy(head(emb), phon[t][y], ignore_index=-1)
                             for t, head in aux.items() if bool((phon[t][y] >= 0).any())]
                    if terms:
                        loss = loss + a.phon_weight * torch.stack(terms).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                d = min(a.ema, (1 + epoch * steps + k) / (10 + epoch * steps + k))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            total += loss.item()
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            val = score(predict_logits(ema, data, rows["val"][0], cfg, False, perm), rows["val"][1])
            # With --trainval the validation clips are training data, so only the last EMA is kept.
            if (not a.trainval and val["top1"] > best) or (a.trainval and epoch == a.epochs):
                best, best_epoch = val["top1"], epoch
                torch.save(dict(state=ema.state_dict(), cfg=cfg, classes=n_cls, face_flip=cache["face_flip"],
                                aux=aux.state_dict() if aux is not None else None), out / "best.pt")
            if epoch % 25 == 0 or epoch == a.epochs:
                print(f"epoch {epoch} loss {total / steps:.3f} val {val['top1']:.1f} best {best:.1f}@{best_epoch} {time.time() - t0:.0f}s", flush=True)

    ema.load_state_dict(torch.load(out / "best.pt", weights_only=False)["state"])
    res = dict(subset=a.subset, params=params, train_clips=len(ids), best_epoch=best_epoch, cfg=cfg, args=vars(a),
               train_seconds=round(time.time() - t0))
    for name in ("val", "test"):
        i, y = rows[name]
        res[name] = score(predict_logits(ema, data, i, cfg, False, perm), y)
        res[name + "_tta"] = score(predict_logits(ema, data, i, cfg, True, perm), y)
    torch.save(predict_logits(ema, data, rows["test"][0], cfg, True, perm).cpu(), out / "test_probs.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("params", "best_epoch", "val", "test", "test_tta", "train_seconds")}), flush=True)


if __name__ == "__main__":
    main()
