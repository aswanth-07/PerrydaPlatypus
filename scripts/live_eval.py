"""Measure the live caption mode: chain held-out WLASL clips into continuous streams (rest frames included)
and run the same spotter rules as android/signcore Spotter.kt over them, frame by frame.

Each stream is scored against its true word sequence by edit distance: word error rate, words found,
false insertions. Spotter settings are chosen on the val split; the test split is reported.

Delay: for every true word the spotter emits, the time from the last frame of that word's clip in the stream to
the frame at which it was emitted (negative: before the sign ended), median and 90th percentile in ms. Only
emissions inside the word's own clip or the next one count.

--set-rule applies the same-sign answer rule as the shipped 2,000-word model does inside its graph (signsets.py,
sets from train+val listings); --set-conf also thresholds the window on its top word's whole-set probability
(Prediction.setConfidence in Kotlin) instead of the word's own.

Usage: live_eval.py <index.json> <kp_dir> <ckpt.pt> [<ckpt.pt> ...] --subset 100 [--streams 60] [--words 6]
       [--set-rule] [--set-conf] [--grid small|wide] [--out f.json]
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
import signsets
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
    """Average of the checkpoints' probabilities; with `rule` = (set_of, counts) the answer rule is applied and
    the set mass of each word is returned beside it."""

    def __init__(self, ckpts, dev="cuda", rule=None):
        self.rule = rule
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
        p = sum(tv.predict_logits(m, data, torch.zeros(1, dtype=torch.long, device=self.dev), cfg, False, self.perm, keep_none=True)
                for m, cfg in self.models) / len(self.models)
        if self.rule is None:
            return p[0].cpu().numpy(), None
        set_of, counts = self.rule
        n = len(set_of)
        q = p[:, :n].double()
        ruled = signsets.answer_rule(q, set_of, counts)
        return ruled[0].float().cpu().numpy(), signsets.set_mass(q, set_of)[0].float().cpu().numpy()


def run_stream(frames, wh, fps, clf, labels, window_s=2.0, stride=5, threshold=0.45, agree=2, min_hands=0.5, set_conf=False):
    """frames: list of per-frame dicts (pose, hands, face, hands_world). Mirrors Spotter.kt.
    Returns the caption as (word, frame index at which it was emitted)."""
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
        p, mass = clf(pts.astype(np.float32), wld.astype(np.float32), seen.astype(np.float32))
        k = int(p.argmax())
        if k >= len(labels):  # the model's "no sign" class won: treat like resting hands
            rested, streak, streak_word = True, 0, None
            continue
        word, conf = labels[k], float(mass[k] if set_conf and mass is not None else p[k])
        if conf >= threshold and word == streak_word:
            streak += 1
        else:
            streak_word = word if conf >= threshold else None
            streak = 1 if streak_word else 0
        if streak >= agree and (word != last or rested):
            caption.append((word, t))
            last, rested = word, False
    return caption


def streams(entries, kp_dir, split, subset, n_streams, n_words, seed):
    rng = np.random.default_rng(seed)
    pool = [(i["video_id"], e["gloss"].upper()) for e in entries[:subset] for i in e["instances"]
            if i["split"] == split and (pathlib.Path(kp_dir) / f"{i['video_id']}.npz").exists()]
    out = []
    for _ in range(n_streams):
        pick = [pool[j] for j in rng.choice(len(pool), n_words, replace=False)]
        frames, truth, bounds, fps, wh = [], [], [], None, None
        for vid, g in pick:
            raw, f, wh_ = load_raw(pathlib.Path(kp_dir) / f"{vid}.npz")
            fps = fps or f
            wh = wh if wh is not None else wh_
            start = len(frames)
            frames += [{k: raw[k][t] for k in raw} for t in range(len(raw["pose"]))]
            truth.append(g)
            bounds.append((start, len(frames)))
        out.append((frames, truth, bounds, fps or 25.0, wh))
    return out


def delays(cap, truth, bounds, fps):
    """ms from the end of each true word's clip to its emission, for words emitted in their clip or the next."""
    out, used = [], set()
    for j, (g, (s, e)) in enumerate(zip(truth, bounds)):
        until = bounds[j + 1][1] if j + 1 < len(bounds) else 10 ** 9
        for n, (w, t) in enumerate(cap):
            if n not in used and w == g and s <= t < until:
                used.add(n)
                out.append(1000.0 * (t + 1 - e) / fps)
                break
    return out


def score(clf, labels, data, **kw):
    errs = words = found = inserted = 0
    lag = []
    for frames, truth, bounds, fps, wh in data:
        timed = run_stream(frames, wh, fps, clf, labels, **kw)
        cap = [w for w, _ in timed]
        lag += delays(timed, truth, bounds, fps)
        errs += edit(cap, truth)
        words += len(truth)
        found += sum(min(cap.count(w), truth.count(w)) for w in set(truth))
        inserted += max(len(cap) - sum(min(cap.count(w), truth.count(w)) for w in set(cap)), 0)
    return dict(wer=round(100 * errs / words, 1), words_found=round(100 * found / words, 1),
                false_per_100_words=round(100 * inserted / words, 1),
                delay_ms_median=round(float(np.median(lag)), 0) if lag else None,
                delay_ms_p90=round(float(np.percentile(lag, 90)), 0) if lag else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--streams", type=int, default=60)
    ap.add_argument("--words", type=int, default=6)
    ap.add_argument("--no-tune", action="store_true", help="only score the default spotter settings")
    ap.add_argument("--set-rule", action="store_true", help="apply the same-sign answer rule, as the shipped model does")
    ap.add_argument("--set-conf", action="store_true", help="threshold on the top word's whole-set probability")
    ap.add_argument("--grid", default="small", choices=["small", "wide"])
    ap.add_argument("--out")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))
    labels = [e["gloss"].upper() for e in entries[:a.subset]]
    rule = None
    if a.set_rule or a.set_conf:
        set_of, _ = signsets.same_sign_sets(entries, a.subset)
        rule = (set_of.cuda(), signsets.wlasl_counts(entries, a.subset).cuda())
    clf = Classifier(a.ckpts, rule=rule)
    test = streams(entries, a.kp_dir, "test", a.subset, a.streams, a.words, 2)
    default = dict(window_s=2.0, stride=5, threshold=0.45, agree=2, set_conf=a.set_conf)
    res = dict(ckpts=a.ckpts, subset=a.subset, set_rule=rule is not None, set_conf=a.set_conf,
               default_settings=default, test_default=score(clf, labels, test, **default))
    if not a.no_tune:
        val = streams(entries, a.kp_dir, "val", a.subset, a.streams // 2, a.words, 1)
        if a.grid == "small":
            grid = [dict(window_s=w, stride=5, threshold=t, agree=g) for w, t, g in itertools.product((1.2, 1.6, 2.0), (0.35, 0.45, 0.6), (1, 2, 3))]
        else:
            grid = [dict(window_s=w, stride=st, threshold=t, agree=g) for w, st, t, g in
                    itertools.product((1.2, 1.5, 2.0), (3, 5), (0.35, 0.45, 0.5, 0.6), (1, 2))]
        for g in grid:
            g["set_conf"] = a.set_conf
        scored = [(g, score(clf, labels, val, **g)) for g in grid]
        best_wer = min(r["wer"] for _, r in scored)
        # Lowest word error rate on val; among settings within 1 point of it, the one with the lowest median delay.
        near = [(g, r) for g, r in scored if r["wer"] <= best_wer + 1.0 and r["delay_ms_median"] is not None]
        best = min(near, key=lambda gr: gr[1]["delay_ms_median"] or 0.0)[0] if near else min(scored, key=lambda gr: gr[1]["wer"])[0]
        res.update(tuned_on_val=best, val_tuned=next(r for g, r in scored if g is best),
                   test_tuned=score(clf, labels, test, **best))
    print(json.dumps(res))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
