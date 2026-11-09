"""Train the keypoint classifier on a WLASL subset and report held-out test accuracy.

The checkpoint is chosen on validation accuracy; test accuracy is only read at the end.

Usage: train_kp.py <index.json> <kp_dir> <out_dir> --subset 100 [--train-ids ids.txt] [--epochs 150] [--width 192]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as fx

MAX_T = 64
INPUT_T = 48


class TBlock(nn.Module):
    def __init__(self, d, dilation, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, 7, padding=3 * dilation, dilation=dilation, groups=d, bias=False),
            nn.BatchNorm1d(d),
            nn.Conv1d(d, 2 * d, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(2 * d, d, 1, bias=False),
            nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SignTCN(nn.Module):
    def __init__(self, classes, width=192, depth=6, drop=0.2, head_drop=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(fx.FEATURES, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.blocks = nn.Sequential(*[TBlock(width, 2 ** (i % 3), drop) for i in range(depth)])
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(2 * width, classes))

    def forward(self, features):
        h = self.blocks(self.stem(features.transpose(1, 2)))
        return self.head(torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1))


def load_cache(index, kp_dir):
    """Normalised keypoints for every extracted clip, cached because 20k small files are slow to open."""
    kp_dir = pathlib.Path(kp_dir)
    cache = kp_dir / "_cache.pt"
    files = sorted(p for p in kp_dir.glob("*.npz"))
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        if len(c["vid"]) == len(files):
            return c
    X = np.zeros((len(files), MAX_T, fx.POINTS, 3), np.float32)
    S = np.zeros((len(files), MAX_T, 3), np.float32)
    L = np.zeros(len(files), np.int64)
    vids = []
    for n, p in enumerate(files):
        d = np.load(p)
        pts, seen = fx.normalise_clip(d["kp"].astype(np.float32), d["wh"])
        X[n, :len(pts)] = pts
        S[n, :len(pts)] = seen
        L[n] = len(pts)
        vids.append(p.stem)
    c = dict(X=torch.from_numpy(X), S=torch.from_numpy(S), L=torch.from_numpy(L), vid=vids)
    torch.save(c, cache)
    return c


def select(cache, entries, subset):
    pos = {v: n for n, v in enumerate(cache["vid"])}
    rows = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for cls, e in enumerate(entries[:subset]):
        for i in e["instances"]:
            if i["video_id"] in pos:
                rows[i["split"]][0].append(pos[i["video_id"]])
                rows[i["split"]][1].append(cls)
    return {k: (torch.tensor(a), torch.tensor(b)) for k, (a, b) in rows.items()}


def warp(X, S, L, length, train):
    """Resample a batch to `length` steps. Training also jitters the window and the speed."""
    b = len(X)
    device = X.device
    grid = torch.linspace(0, 1, length, device=device).unsqueeze(0).expand(b, -1)
    if train:
        start = torch.rand(b, 1, device=device) * 0.12
        end = 1 - torch.rand(b, 1, device=device) * 0.12
        power = torch.empty(b, 1, device=device).uniform_(0.75, 1.3)
        grid = start + (end - start) * grid.pow(power)
    pos = grid * (L.unsqueeze(1).float() - 1).clamp_min(0)
    lo = pos.floor().long()
    hi = torch.min(lo + 1, (L.unsqueeze(1) - 1).clamp_min(0))
    w = (pos - lo.float())
    rows = torch.arange(b, device=device).unsqueeze(1)
    wx = w[:, :, None, None]
    out_x = (1 - wx) * X[rows, lo] + wx * X[rows, hi]
    ws = w[:, :, None]
    return out_x, (1 - ws) * S[rows, lo] + ws * S[rows, hi]


def single_frame(X, S, L, train):
    """One frame per clip, repeated over time: all a frame-only model would get to see."""
    b = len(X)
    device = X.device
    pos = torch.empty(b, device=device).uniform_(0.3, 0.7) if train else torch.full((b,), 0.5, device=device)
    idx = (pos * (L.float() - 1).clamp_min(0)).round().long()
    rows = torch.arange(b, device=device)
    return X[rows, idx].unsqueeze(1).expand(-1, INPUT_T, -1, -1), S[rows, idx].unsqueeze(1).expand(-1, INPUT_T, -1)


def augment(x, s):
    b, t = x.shape[:2]
    device = x.device
    x = x.clone()
    s = s.clone()
    # Mirror: x -> -x with hands and lateral body points swapped, which also covers left-handed signers.
    flip = torch.rand(b, device=device) < 0.5
    if flip.any():
        m = x[flip]
        m[..., 0] *= -1
        idx = torch.arange(fx.POINTS, device=device)
        for p, q in fx.FLIP_PAIRS:
            idx[p], idx[q] = q, p
        for k in range(21):
            idx[fx.LEFT[0] + k], idx[fx.RIGHT[0] + k] = fx.RIGHT[0] + k, fx.LEFT[0] + k
        x[flip] = m[:, :, idx]
        ss = s[flip]
        s[flip] = ss[:, :, [0, 2, 1]]
    # Camera changes: in-plane roll, a yaw-like x squeeze, scale, shift.
    angle = torch.empty(b, device=device).uniform_(-0.26, 0.26)
    cos, sin = angle.cos()[:, None, None], angle.sin()[:, None, None]
    px, py = x[..., 0].clone(), x[..., 1].clone()
    x[..., 0] = cos * px - sin * py
    x[..., 1] = sin * px + cos * py
    x[..., 0] *= torch.empty(b, 1, 1, device=device).uniform_(0.75, 1.1)
    x *= torch.empty(b, 1, 1, 1, device=device).uniform_(0.85, 1.2)
    x += torch.empty(b, 1, 1, 3, device=device).uniform_(-0.3, 0.3)
    x += torch.randn_like(x) * 0.01
    # Occlusion: lose a hand or random points for a stretch of frames.
    for side, rng in ((1, fx.LEFT), (2, fx.RIGHT)):
        hit = (torch.rand(b, device=device) < 0.15).nonzero().flatten()
        for k in hit.tolist():
            a = int(torch.randint(0, t, (1,)))
            n = int(torch.randint(4, t // 2, (1,)))
            x[k, a:a + n, rng[0]:rng[1]] = 0
            s[k, a:a + n, side] = 0
    drop = torch.rand(b, 1, fx.POINTS, device=device) < 0.05
    x = x.masked_fill(drop.unsqueeze(-1), 0)
    return x.clamp(-fx.CLAMP, fx.CLAMP), s


@torch.no_grad()
def evaluate(model, data, rows, device, frames=None, single=False):
    model.eval()
    ids, y = rows
    top1 = top5 = 0
    per_class = {}
    for chunk, yc in zip(ids.split(256), y.split(256)):
        X, S, L = data["X"][chunk.to(device)], data["S"][chunk.to(device)], data["L"][chunk.to(device)]
        if frames is not None:  # keep only `frames` evenly spaced source frames, as a faster extractor would
            X, S = warp(X, S, L, frames, False)
            L = torch.full_like(L, frames)
        x, s = single_frame(X, S, L, False) if single else warp(X, S, L, INPUT_T, False)
        logits = model(fx.make_features_torch(x, s))
        yc = yc.to(device)
        order = logits.topk(min(5, logits.shape[1]), dim=1).indices
        hit1 = order[:, 0] == yc
        top1 += hit1.sum().item()
        top5 += (order == yc[:, None]).any(dim=1).sum().item()
        for c, h in zip(yc.tolist(), hit1.tolist()):
            per_class.setdefault(c, []).append(h)
    n = max(len(y), 1)
    return dict(n=len(y), top1=100 * top1 / n, top5=100 * top5 / n,
                class_avg=100 * float(np.mean([np.mean(v) for v in per_class.values()])) if per_class else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--train-ids")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--single-frame", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = json.load(open(a.index, encoding="utf-8"))
    cache = load_cache(entries, a.kp_dir)
    rows = select(cache, entries, a.subset)
    if a.train_ids:
        keep = set(pathlib.Path(a.train_ids).read_text().split())
        mask = torch.tensor([cache["vid"][i] in keep for i in rows["train"][0].tolist()])
        rows["train"] = (rows["train"][0][mask], rows["train"][1][mask])
    data = {k: cache[k].to(device) for k in ("X", "S", "L")}
    print({k: len(v[0]) for k, v in rows.items()}, flush=True)

    model = SignTCN(a.subset, a.width, a.depth).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    ids, labels = rows["train"]
    steps = max(len(ids) // a.batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.1)
    best, best_epoch = -1.0, 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = torch.randperm(len(ids))
        total = 0.0
        for k in range(steps):
            pick = order[k * a.batch:(k + 1) * a.batch]
            c = ids[pick].to(device)
            pick_frames = single_frame if a.single_frame else lambda X, S, L, train: warp(X, S, L, INPUT_T, train)
            x, s = augment(*pick_frames(data["X"][c], data["S"][c], data["L"][c], True))
            loss = F.cross_entropy(model(fx.make_features_torch(x, s)), labels[pick].to(device), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == a.epochs:
            val = evaluate(model, data, rows["val"], device, single=a.single_frame)
            if val["top1"] > best:
                best, best_epoch = val["top1"], epoch
                torch.save(dict(state=model.state_dict(), classes=a.subset, width=a.width, depth=a.depth), out / "best.pt")
            if epoch % 25 == 0 or epoch == a.epochs:
                print(f"epoch {epoch} loss {total / steps:.3f} val {val['top1']:.1f} best {best:.1f}@{best_epoch} {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["state"])
    result = dict(subset=a.subset, params=params, train_clips=len(ids), best_epoch=best_epoch,
                  single_frame=a.single_frame,
                  val=evaluate(model, data, rows["val"], device, single=a.single_frame),
                  test=evaluate(model, data, rows["test"], device, single=a.single_frame),
                  train_seconds=round(time.time() - t0))
    if not a.single_frame:
        result["test_vs_frames"] = {str(f): evaluate(model, data, rows["test"], device, frames=f)["top1"] for f in (4, 8, 12, 16, 24, 32)}
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
