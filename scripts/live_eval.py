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

--rate-div k replays every k-th frame (a tracker at fps/k); --fixed scores given settings on test only.

--segment scores the boundary spotter (Spotter.kt, boundary mode) instead of the sliding window alone: a sign runs
from the frame a wrist rises above the rest line to the frame both have been back below it for rest_ms, and that
whole stretch is classified once, like a clip. Past max_one_s without a rest the sliding window captions.
Chosen on val streams and reported on test, 15 Nov 2026 (2,000 words, soup_mix16 with the answer rule): word
error rate 49.2 -> 42.8 on the 60 test streams (runs/analysis_2026-11-15).

--compare scores both spotters on the same test streams (--seed, --streams) and gives the paired gain in word
error rate with a bootstrap interval over streams; the sliding window runs with --sliding settings (default: the
ones that won on val streams, window_s=1.5,stride=5,threshold=0.35,agree=2).

Usage: live_eval.py <index.json> <kp_dir> <ckpt.pt> [<ckpt.pt> ...] --subset 100 [--streams 60] [--words 6]
       [--set-rule] [--set-conf] [--grid small|wide] [--rate-div k] [--fixed window_s=1.5,stride=5,...] [--out f.json]
       [--segment] [--compare [--sliding window_s=1.5,...] [--seed 2]]
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


def wrist_height(f, wh, sw):
    """Height of the higher wrist above the shoulder line in shoulder widths (negative: below), and the updated
    running shoulder width; (None, sw) when the body is not seen. Mirrors Spotter.wristHeight in Spotter.kt."""
    pose = f["pose"]
    if not np.isfinite(pose[11, 0]) or not np.isfinite(pose[12, 0]):
        return None, sw
    w, h = float(wh[0]), float(wh[1])
    side = max(w, h)
    ax, ay = w / side, h / side
    cur = float(np.hypot((pose[11, 0] - pose[12, 0]) * ax, (pose[11, 1] - pose[12, 1]) * ay))
    sw = cur if sw is None else 0.9 * sw + 0.1 * cur
    mid = (pose[11, 1] + pose[12, 1]) / 2
    hs = [(mid - pose[k, 1]) * ay / max(sw, 1e-3) for k in (15, 16) if np.isfinite(pose[k, 1])]
    return (max(hs) if hs else None), sw


def rest_frames(frames, n):
    """n frames with nobody in view, appended to a stream so that its last stretch closes as it would live."""
    blank = {k: np.full_like(v, np.nan) for k, v in frames[0].items()}
    return frames + [blank] * n


def run_stream_seg(frames, wh, fps, clf, labels, rest_h=-1.0, rest_ms=200, pre_ms=300, threshold=0.15, min_ms=400,
                   min_hands=0.3, max_one_s=4.0, set_conf=False, window_s=1.5, stride=5, slide_threshold=0.35, agree=2,
                   preview=True, memo=None):
    """The boundary spotter, line for line what Spotter.kt does in boundary mode (pushBoundary and close), on
    millisecond timestamps. One classification per stretch between two rests; the sliding window only for a
    stretch longer than max_s, and (with preview) as the fallback when a stretch closes below threshold.
    memo: a dict that caches classifications of this stream across calls (a settings grid asks for the same
    windows again and again). Returns the caption as (word, frame index at which it was emitted)."""
    n = len(frames)
    ts = [int(round(1000.0 * t / fps)) for t in range(n)]
    window_ms, max_one_ms = int(round(window_s * 1000)), int(round(max_one_s * 1000))
    seen = np.concatenate([[0], np.cumsum([1 if (np.isfinite(f["hands"][0, 0, 0]) or np.isfinite(f["hands"][1, 0, 0])) else 0
                                            for f in frames])])
    memo = {} if memo is None else memo

    def classify(a, b):
        if (a, b) not in memo:
            w = frames[a:b]
            st = {k: np.stack([f[k] for f in w]) for k in w[0]}
            pts, wld, sn = dv.normalise(st["pose"], st["hands"], st["face"], st["hands_world"], wh)
            p, mass = clf(pts.astype(np.float32), wld.astype(np.float32), sn.astype(np.float32))
            k = int(p.argmax())
            memo[(a, b)] = (None, 0.0, 0.0) if k >= len(labels) else (labels[k], float(p[k]), float(mass[k]) if mass is not None else float(p[k]))
        r = memo[(a, b)]
        return r[0], r[2 if set_conf else 1]

    def hand_fraction(a, b):
        return (seen[b] - seen[a]) / max(b - a, 1)

    def first_from(t, since_ms):          # index of the first frame at or after since_ms, looking back from t
        i = t
        while i > 0 and ts[i - 1] >= since_ms:
            i -= 1
        return i

    def window_start(t, first):           # the sliding window: the last window_ms, never before the stretch
        i = t
        while i > first and ts[t] - ts[i - 1] < window_ms:
            i -= 1
        return i

    caption = []
    sw = None
    open_at, low_since, cont, pending = None, -1, False, []
    streak_word, streak, last, since = None, 0, None, 0
    for t in range(n):
        h, sw = wrist_height(frames[t], wh, sw)
        up = h is not None and h > rest_h + (0.0 if open_at is not None else 0.1)   # a little hysteresis to open
        if open_at is None:
            if up:
                open_at, low_since, cont, pending = ts[t] - pre_ms, -1, False, []
                streak_word, streak, last, since = None, 0, None, 0
            continue
        if up:
            low_since = -1
        elif low_since < 0:
            low_since = ts[t]
        if low_since >= 0 and ts[t] - low_since >= rest_ms:      # close(): the hands are back at rest
            first, was_cont, up_ms = first_from(t, open_at), cont, low_since - open_at
            open_at, low_since, cont, since = None, -1, False, 0
            if was_cont:
                a = window_start(t, first)
                if hand_fraction(a, t + 1) >= 0.5:
                    word, conf = classify(a, t + 1)
                    if word is not None and conf >= slide_threshold and word != last:
                        caption.append((word, t))
                continue
            if up_ms < min_ms or hand_fraction(first, t + 1) < min_hands:
                pending = []
                continue
            word, conf = classify(first, t + 1)
            if word is not None and conf >= threshold:
                caption.append((word, t))
            else:
                caption += [(w_, t) for w_ in pending]
            pending = []
            continue
        since += 1
        first = first_from(t, open_at)
        if since >= stride and (preview or cont) and t + 1 - first >= 8:
            since = 0
            a = window_start(t, first)
            if hand_fraction(a, t + 1) >= 0.5:
                word, conf = classify(a, t + 1)
                sure = word is not None and conf >= slide_threshold
                if sure and word == streak_word:
                    streak += 1
                else:
                    streak_word = word if sure else None
                    streak = 1 if streak_word else 0
                if streak >= agree and word != last:
                    last = word
                    if cont:
                        caption.append((word, t))
                    else:
                        pending.append(word)
        if not cont and ts[t] - open_at >= max_one_ms:           # no rest for this long: caption as they come
            cont = True
            caption += [(w_, t) for w_ in pending]
            pending = []
    return caption


def streams(entries, kp_dir, split, subset, n_streams, n_words, seed, rate_div=1, keep=None):
    """keep: a predicate on the video id, to draw the streams from part of the split only."""
    rng = np.random.default_rng(seed)
    pool = [(i["video_id"], e["gloss"].upper()) for e in entries[:subset] for i in e["instances"]
            if i["split"] == split and (pathlib.Path(kp_dir) / f"{i['video_id']}.npz").exists()
            and (keep is None or keep(i["video_id"]))]
    out = []
    for _ in range(n_streams):
        pick = [pool[j] for j in rng.choice(len(pool), n_words, replace=False)]
        frames, truth, bounds, fps, wh = [], [], [], None, None
        for vid, g in pick:
            raw, f, wh_ = load_raw(pathlib.Path(kp_dir) / f"{vid}.npz")
            if rate_div > 1:   # a slower tracker: every rate_div-th frame
                raw = {k: v[::rate_div] for k, v in raw.items()}
                f = f / rate_div
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


MEMO = {}   # classifications of each stream's windows, kept across the settings of a grid (boundary spotter)


def score(clf, labels, data, segment=False, per_stream=None, **kw):
    """per_stream: a list that receives each stream's edit distance, for paired comparisons."""
    errs = words = found = inserted = 0
    lag = []
    for frames, truth, bounds, fps, wh in data:
        if segment:   # half a second of nobody in view closes the last stretch, as the signer stepping away does
            timed = run_stream_seg(rest_frames(frames, int(round(fps / 2))), wh, fps, clf, labels,
                                   memo=MEMO.setdefault(id(frames), {}), **kw)
        else:
            timed = run_stream(frames, wh, fps, clf, labels, **kw)
        cap = [w for w, _ in timed]
        lag += delays(timed, truth, bounds, fps)
        errs += edit(cap, truth)
        if per_stream is not None:
            per_stream.append(edit(cap, truth))
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
    ap.add_argument("--rate-div", type=int, default=1, help="keep every k-th frame: tracking at fps/k")
    ap.add_argument("--fixed", help="score only these settings on test, e.g. window_s=1.5,stride=5,threshold=0.35,agree=2")
    ap.add_argument("--segment", action="store_true", help="score the boundary spotter instead of the sliding window")
    ap.add_argument("--compare", action="store_true", help="sliding window against the boundary spotter on the same test streams")
    ap.add_argument("--sliding", default="window_s=1.5,stride=5,threshold=0.35,agree=2", help="sliding-window settings for --compare")
    ap.add_argument("--seed", type=int, default=2, help="seed of the test streams")
    ap.add_argument("--out")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))
    labels = [e["gloss"].upper() for e in entries[:a.subset]]
    rule = None
    if a.set_rule or a.set_conf:
        set_of, _ = signsets.same_sign_sets(entries, a.subset)
        rule = (set_of.cuda(), signsets.wlasl_counts(entries, a.subset).cuda())
    clf = Classifier(a.ckpts, rule=rule)
    test = streams(entries, a.kp_dir, "test", a.subset, a.streams, a.words, a.seed, a.rate_div)
    if a.compare:
        slide = dict(set_conf=a.set_conf)
        for kv in a.sliding.split(","):
            k, v = kv.split("=")
            slide[k] = int(v) if k in ("stride", "agree") else float(v)
        e_slide, e_seg = [], []
        res = dict(ckpts=a.ckpts, subset=a.subset, set_rule=rule is not None, streams=a.streams, words=a.words, seed=a.seed,
                   sliding_settings=slide, sliding=score(clf, labels, test, per_stream=e_slide, **slide),
                   boundary=score(clf, labels, test, True, per_stream=e_seg, set_conf=a.set_conf))
        d = np.array(e_slide, float) - np.array(e_seg, float)
        rng = np.random.default_rng(0)
        boots = [100 * d[rng.integers(0, len(d), len(d))].sum() / (a.words * len(d)) for _ in range(4000)]
        res["wer_gain_points"] = round(100 * float(d.sum()) / (a.words * len(d)), 1)
        res["wer_gain_ci95"] = [round(float(x), 1) for x in np.percentile(boots, [2.5, 97.5])]
        print(json.dumps(res))
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        return
    default = dict(window_s=2.0, stride=5, threshold=0.45, agree=2, set_conf=a.set_conf)
    if a.segment:
        default = dict(rest_h=-1.0, rest_ms=200, max_one_s=4.0, threshold=0.15, preview=True, set_conf=a.set_conf)
    if a.fixed:
        default = dict(set_conf=a.set_conf)
        for kv in a.fixed.split(","):
            k, v = kv.split("=")
            default[k] = int(v) if k in ("stride", "agree", "rest_ms", "pre_ms") else v == "1" if k == "preview" else float(v)
        a.no_tune = True
    res = dict(ckpts=a.ckpts, subset=a.subset, set_rule=rule is not None, set_conf=a.set_conf, rate_div=a.rate_div,
               segment=a.segment, default_settings=default, test_default=score(clf, labels, test, a.segment, **default))
    if not a.no_tune:
        val = streams(entries, a.kp_dir, "val", a.subset, a.streams // 2, a.words, 1)
        if a.segment:
            grid = [dict(rest_h=h, rest_ms=r, max_one_s=m, threshold=t, preview=pv) for h, r, m, t, pv in
                    itertools.product((-1.1, -1.0, -0.9), (120, 200, 280), (3.2, 4.0), (0.0, 0.15, 0.3), (True, False))]
        elif a.grid == "small":
            grid = [dict(window_s=w, stride=5, threshold=t, agree=g) for w, t, g in itertools.product((1.2, 1.6, 2.0), (0.35, 0.45, 0.6), (1, 2, 3))]
        else:
            grid = [dict(window_s=w, stride=st, threshold=t, agree=g) for w, st, t, g in
                    itertools.product((1.2, 1.5, 2.0), (3, 5), (0.35, 0.45, 0.5, 0.6), (1, 2))]
        for g in grid:
            g["set_conf"] = a.set_conf
        scored = [(g, score(clf, labels, val, a.segment, **g)) for g in grid]
        best_wer = min(r["wer"] for _, r in scored)
        # Lowest word error rate on val; among settings within 1 point of it, the one with the lowest median delay.
        near = [(g, r) for g, r in scored if r["wer"] <= best_wer + 1.0 and r["delay_ms_median"] is not None]
        best = min(near, key=lambda gr: gr[1]["delay_ms_median"] or 0.0)[0] if near else min(scored, key=lambda gr: gr[1]["wer"])[0]
        res.update(tuned_on_val=best, val_tuned=next(r for g, r in scored if g is best),
                   test_tuned=score(clf, labels, test, a.segment, **best))
    print(json.dumps(res))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
