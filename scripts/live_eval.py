"""Measure the live caption mode: chain held-out WLASL clips into continuous streams (rest frames included)
and run the same spotter rules as android/signcore Spotter.kt over them, frame by frame.

Each stream is scored against its true word sequence by edit distance: word error rate, words found,
false insertions. Spotter settings are chosen on the val split; the test split is reported.

Usage: live_eval.py <index.json> <kp_dir> <ckpt.pt> [<ckpt.pt> ...] --subset 100 [--streams 60] [--words 6]
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_v2 as dv
import model_v2 as mv
import train_v2 as tv


def load_raw(path):
    d = np.load(path)
    return {k: d[k].astype(np.float64) for k in ("pose", "hands", "face", "hands_world")}, float(d["fps"]), d["wh"]


def edit(a, b):
    dp = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, dp[j] = dp[j], cur
    return int(dp[-1])


class Classifier:
    def __init__(self, ckpts, dev="cuda"):
        self.models = []
        for c in ckpts:
            ck = torch.load(c, weights_only=False)
            m = mv.build(ck["cfg"], ck["classes"]).to(dev).eval()
            m.load_state_dict(ck["state"])
            self.models.append((m, ck["cfg"]))
        self.perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).to(dev)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, pts, wld, seen):
        X = torch.from_numpy(pts)[None].to(self.dev)
        W = torch.from_numpy(wld)[None].to(self.dev)
        S = torch.from_numpy(seen)[None].to(self.dev)
        L = torch.tensor([len(pts)], device=self.dev)
        data = dict(X=X, W=W, S=S, L=L)
        return sum(tv.predict_logits(m, data, torch.zeros(1, dtype=torch.long, device=self.dev), cfg, False, self.perm, keep_none=True)
                   for m, cfg in self.models)[0].cpu().numpy() / len(self.models)


def run_stream(frames, wh, fps, clf, labels, window_s=2.0, stride=5, threshold=0.45, agree=2, min_hands=0.5):
    """frames: list of per-frame dicts (pose, hands, face, hands_world). Mirrors Spotter.kt."""
    win = max(int(round(window_s * fps)), 8)
    caption, streak_word, streak, last, rested = [], None, 0, None, True
    since = 0
    for t in range(len(frames)):
        since += 1
        if since < stride or t + 1 < 8:
            continue
        since = 0
        w = frames[max(0, t + 1 - win): t + 1]
        hands = sum(1 for f in w if np.isfinite(f["hands"][0, 0, 0]) or np.isfinite(f["hands"][1, 0, 0]))
        if hands < min_hands * len(w):
            rested, streak, streak_word = True, 0, None
            continue
        st = {k: np.stack([f[k] for f in w]) for k in w[0]}
        pts, wld, seen = dv.normalise(st["pose"], st["hands"], st["face"], st["hands_world"], wh)
        p = clf(pts.astype(np.float32), wld.astype(np.float32), seen.astype(np.float32))
        k = int(p.argmax())
        if k >= len(labels):  # the model's "no sign" class won: treat like resting hands
            rested, streak, streak_word = True, 0, None
            continue
        word, conf = labels[k], float(p[k])
        if conf >= threshold and word == streak_word:
            streak += 1
        else:
            streak_word = word if conf >= threshold else None
            streak = 1 if streak_word else 0
        if streak >= agree and (word != last or rested):
            caption.append(word)
            last, rested = word, False
    return caption


def streams(entries, kp_dir, split, subset, n_streams, n_words, seed):
    rng = np.random.default_rng(seed)
    pool = [(i["video_id"], e["gloss"].upper()) for e in entries[:subset] for i in e["instances"]
            if i["split"] == split and (pathlib.Path(kp_dir) / f"{i['video_id']}.npz").exists()]
    out = []
    for _ in range(n_streams):
        pick = [pool[j] for j in rng.choice(len(pool), n_words, replace=False)]
        frames, truth, fps, wh = [], [], None, None
        for vid, g in pick:
            raw, f, wh_ = load_raw(pathlib.Path(kp_dir) / f"{vid}.npz")
            fps = fps or f
            wh = wh if wh is not None else wh_
            frames += [{k: raw[k][t] for k in raw} for t in range(len(raw["pose"]))]
            truth.append(g)
        out.append((frames, truth, fps or 25.0, wh))
    return out


def score(clf, labels, data, **kw):
    errs = words = found = inserted = 0
    for frames, truth, fps, wh in data:
        cap = run_stream(frames, wh, fps, clf, labels, **kw)
        errs += edit(cap, truth)
        words += len(truth)
        found += sum(min(cap.count(w), truth.count(w)) for w in set(truth))
        inserted += max(len(cap) - sum(min(cap.count(w), truth.count(w)) for w in set(cap)), 0)
    return dict(wer=round(100 * errs / words, 1), words_found=round(100 * found / words, 1),
                false_per_100_words=round(100 * inserted / words, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--streams", type=int, default=60)
    ap.add_argument("--words", type=int, default=6)
    ap.add_argument("--no-tune", action="store_true", help="only score the default spotter settings")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))
    labels = [e["gloss"].upper() for e in entries[:a.subset]]
    clf = Classifier(a.ckpts)
    test = streams(entries, a.kp_dir, "test", a.subset, a.streams, a.words, 2)
    default = dict(window_s=2.0, stride=5, threshold=0.45, agree=2)
    if a.no_tune:
        print(json.dumps(dict(default_settings=default, test_default=score(clf, labels, test, **default))))
        return
    val = streams(entries, a.kp_dir, "val", a.subset, a.streams // 2, a.words, 1)
    grid = [dict(window_s=w, stride=5, threshold=t, agree=g) for w, t, g in itertools.product((1.2, 1.6, 2.0), (0.35, 0.45, 0.6), (1, 2, 3))]
    best = min(grid, key=lambda kw: score(clf, labels, val, **kw)["wer"])
    print(json.dumps(dict(default_settings=default, test_default=score(clf, labels, test, **default),
                          tuned_on_val=best, test_tuned=score(clf, labels, test, **best))))


if __name__ == "__main__":
    main()
