"""Measure live captions on continuous signing: signs made one after another without lowering the hands.

Streams are made from held-out WLASL clips by contsign.cont_streams, in three kinds:
  norest  the hands never rest: a join is a direct movement (50%), a movement with a pose held in the air for
          0.3-1.5 s (30%), or a wait of 0.5-1.5 s at a relaxed pose in the air (20%)
  mixed   direct 40%, held pose 25%, relaxed wait 15%, a real rest 20%
  rest    whole clips chained as they are (scripts/live_eval.py's streams): the hands drop after most signs

Two spotters are scored on the same streams by word error rate against the true word sequence:
  boundary    Spotter.kt as shipped on 15 Nov (live_eval.run_stream_seg): one answer per stretch between two
              rests, the sliding window after 4 s without one
  continuous  a window of --window seconds classified every `stride` frames while the hands are up; a run of
              windows that name a sign is one sign, captioned when it ends (contsign.Decoder); meant for a model
              trained on such windows (train.py --cont)

Decoder settings are chosen without the clips they are scored on. The shipped models train on WLASL train and
val, so val streams would flatter every threshold; instead the test clips are split in two halves by video id,
streams are drawn from each half, the settings that are best on one half are scored on the other, and the two
scores are pooled ("cross-fitted"). The settings to ship are the best on both halves together. A setting wins
on the mean word error rate over the three kinds; among settings within one point of the best, the one with the
lowest median delay.

Usage: live_cont.py <index.json> <kp_dir> <ckpt.pt> [...] [--subset 2000] [--base <ckpt.pt> ...] [--streams 60]
       [--words 6] [--window 1.2] [--no-rule] [--fixed stride=2,lo=0.1,...] [--onnx model.onnx --sets sets.json]
       [--out f.json]
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import contsign as cs
import normalise as dv
import live_eval as le
import model as mv
import signsets
import train as tv

KINDS = dict(norest=(0.5, 0.3, 0.0, 0.2), mixed=(0.4, 0.25, 0.2, 0.15))   # shares of (direct, pause, rest, hover) joins
TWIN = None            # {word: same-sign set id}, set by main() when the vocabulary has sets
PRE_TAIL_MS = 1000      # windows are classified this long after a rest closes; the decoder's tail is at most this


class WindowModel:
    """Checkpoints averaged; returns per window the word probabilities (after the same-sign answer rule when
    `rule`), each word's whole-set probability, and the probability of "no sign" (0 for a model without it)."""

    def __init__(self, ckpts, entries, subset, rule=True, dev="cuda"):
        self.models, self.n, self.dev = [], subset, dev
        for c in ckpts:
            ck = torch.load(c, weights_only=False)
            m = mv.build(ck["cfg"], ck["classes"]).to(dev).eval()
            m.load_state_dict(ck["state"])
            self.models.append((m, ck["cfg"]))
        self.perm = tv.flip_perm(ck.get("face_flip", torch.arange(62))).to(dev)
        self.set_of = self.counts = None
        if rule:
            self.set_of = signsets.same_sign_sets(entries, subset)[0].to(dev)
            self.counts = signsets.wlasl_counts(entries, subset).to(dev)

    @torch.no_grad()
    def __call__(self, wins):
        n, t = len(wins), max(len(w[0]) for w in wins)
        X = torch.zeros(n, t, dv.P, 3)
        W = torch.zeros(n, t, 2, 21, 3)
        S = torch.zeros(n, t, 4)
        L = torch.zeros(n, dtype=torch.long)
        for k, (pts, wld, seen) in enumerate(wins):
            X[k, :len(pts)], W[k, :len(pts)], S[k, :len(pts)], L[k] = torch.from_numpy(pts), torch.from_numpy(wld), torch.from_numpy(seen), len(pts)
        data = dict(X=X.to(self.dev), W=W.to(self.dev), S=S.to(self.dev), L=L.to(self.dev))
        ids = torch.arange(n, device=self.dev)
        p = 0
        for m, cfg in self.models:
            q = tv.predict_logits(m, data, ids, cfg, False, self.perm, keep_none=True)
            if q.shape[1] == self.n:
                q = torch.cat([q, torch.zeros_like(q[:, :1])], 1)
            p = p + q[:, :self.n + 1] / len(self.models)
        words, none = p[:, :self.n].double(), p[:, self.n]
        if self.set_of is not None:
            mass = signsets.set_mass(words, self.set_of)
            words = signsets.answer_rule(words, self.set_of, self.counts)
        else:
            mass = words
        return words.float().cpu().numpy().astype(np.float16), mass.float().cpu().numpy().astype(np.float16), none.float().cpu().numpy()


def gate_series(frames, wh, fps, rest_ms=200, tail_ms=PRE_TAIL_MS):
    gate, sw, hs, act = cs.RestGate(rest_ms=rest_ms, tail_ms=tail_ms), None, [], []
    for t, f in enumerate(frames):
        h, sw = le.wrist_height(f, wh, sw)
        hs.append(h)
        act.append(gate.push(int(round(1000.0 * t / fps)), h))
    return hs, act


def posteriors(frames, wh, fps, model, window_s, min_hand_frames=6):
    """Window answers for every frame at which the phone might classify: {frame: (probs, mass, none)}."""
    n, wn = len(frames), max(int(round(window_s * fps)), 8)
    hand = np.concatenate([[0], np.cumsum([1 if (np.isfinite(f["hands"][0, 0, 0]) or np.isfinite(f["hands"][1, 0, 0])) else 0 for f in frames])])
    hs, act = gate_series(frames, wh, fps)
    idx, wins = [], []
    for t in range(n):
        a = max(0, t + 1 - wn)
        if not act[t] or t + 1 - a < 8 or hand[t + 1] - hand[a] < min_hand_frames:
            continue
        w = frames[a:t + 1]
        st = {k: np.stack([f[k] for f in w]) for k in w[0]}
        wins.append(dv.normalise(st["pose"], st["hands"], st["face"], st["hands_world"], wh))
        idx.append(t)
    if not wins:
        return {}, hs
    out = {}
    for lo in range(0, len(wins), 1024):
        probs, mass, none = model(wins[lo:lo + 1024])
        for k, t in enumerate(idx[lo:lo + 1024]):
            out[t] = (cs.top_answers(probs[k], mass[k]), float(none[k]))
    return out, hs


def run_cont(n, fps, post, hs, labels, set_of, stride=2, lo=0.1, gap=1, emin=2.5, commit=0.0, rest_ms=200, tail_ms=600):
    """The continuous spotter over one stream, frame by frame, as Spotter.kt does it. Returns (word, frame)."""
    gate, dec = cs.RestGate(rest_ms=rest_ms, tail_ms=tail_ms), cs.Decoder(set_of, lo, gap, emin, commit)
    since, caption = -1, []
    for t in range(n):
        if not gate.push(int(round(1000.0 * t / fps)), hs[t]):
            if since >= 0:                              # the hands came to rest: the open sign is settled
                k = dec.flush()
                if k is not None:
                    caption.append((labels[k], t))
            since = -1
            continue
        since = stride if since < 0 else since + 1      # the first active frame classifies at once
        if since < stride:
            continue
        since = 0
        r = post.get(t)
        k = dec.push(*r) if r is not None else dec.push(None, 1.0)
        if k is not None:
            caption.append((labels[k], t))
    return caption


def tally(caps, data):
    """Counts over streams: errors (edit distance), true words, words found, false words, and the delays."""
    errs = words = found = inserted = same = 0
    lag, per = [], []
    for timed, (_frames, truth, bounds, fps, *_rest) in zip(caps, data):
        cap = [w for w, _ in timed]
        lag += le.delays(timed, truth, bounds, fps)
        e = le.edit(cap, truth)
        per.append(e)
        errs += e
        if TWIN is not None:    # a word that is the same sign as a true word of the stream counts as that word
            same += le.edit([next((w2 for w2 in truth if TWIN.get(w2) == TWIN.get(w, w)), w) for w in cap], truth)
        words += len(truth)
        found += sum(min(cap.count(w), truth.count(w)) for w in set(truth))
        inserted += max(len(cap) - sum(min(cap.count(w), truth.count(w)) for w in set(cap)), 0)
    return dict(errs=errs, words=words, found=found, inserted=inserted, lag=lag, per=per, same=same)


def report(*tallies):
    """One or more tallies pooled into rates."""
    errs, words = sum(t["errs"] for t in tallies), sum(t["words"] for t in tallies)
    lag = [v for t in tallies for v in t["lag"]]
    return dict(wer=round(100 * errs / words, 1), wer_same_sign=round(100 * sum(t.get('same', 0) for t in tallies) / words, 1) if TWIN is not None else None,
                words_found=round(100 * sum(t["found"] for t in tallies) / words, 1),
                false_per_100_words=round(100 * sum(t["inserted"] for t in tallies) / words, 1),
                delay_ms_median=round(float(np.median(lag)), 0) if lag else None,
                delay_ms_p90=round(float(np.percentile(lag, 90)), 0) if lag else None, streams=sum(len(t["per"]) for t in tallies))


def half(video_id):
    """Which half of the test clips a video belongs to (a hash of its id, so the halves share no clip)."""
    import zlib
    return zlib.crc32(str(video_id).encode()) % 2


def make_streams(entries, kp_dir, split, subset, n, words, seed, keep=None):
    """{kind: streams}; each stream ends with half a second of nobody in view, as a signer stepping away does."""
    out = {}
    for kind, mix in KINDS.items():
        out[kind] = cs.cont_streams(entries, kp_dir, split, subset, n, words, seed, mix, keep)
    out["rest"] = le.streams(entries, kp_dir, split, subset, n, words, seed, keep=keep)
    for kind in out:
        out[kind] = [(le.rest_frames(s[0], int(round(s[3] / 2))),) + tuple(s[1:]) for s in out[kind]]
    return out


class OnnxWindowModel:
    """The exported INT8 graph as the phone runs it (probs, set_probs and, for a live model, none).
    half: keep the answers as float16 (memory, for whole test sets); False keeps the graph's own float32."""

    def __init__(self, path, t_in=64, half=True):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.sess.get_outputs()]
        self.t_in, self.set_of, self.counts = t_in, None, None
        self.dtype = np.float16 if half else np.float32

    def __call__(self, wins):
        def rs(a):
            n = len(a)
            pos = np.linspace(0, 1, self.t_in) * max(n - 1, 0)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, max(n - 1, 0))
            w = (pos - lo).astype(np.float32).reshape((-1,) + (1,) * (a.ndim - 1))
            return (a[lo] + (a[hi] - a[lo]) * w)[None].astype(np.float32)

        want = ["probs", "set_probs"] + (["none"] if "none" in self.names else [])
        P, M, N = [], [], []
        for pts, wld, seen in wins:
            r = self.sess.run(want, dict(points=rs(pts), world=rs(wld), seen=rs(seen)))
            P.append(r[0][0])
            M.append(r[1][0])
            N.append(float(r[2][0, 0]) if len(r) > 2 else 0.0)
        return np.stack(P).astype(self.dtype), np.stack(M).astype(self.dtype), np.array(N, np.float32)


# Every fourth frame of 25 fps: about six windows a second, the rate the phone's power was measured at; strides 1 to 4
# score within a point of each other (runs/analysis_2026-11-17/seg_full_m_s*.json).
GRID = [dict(stride=4, lo=lo, gap=g, emin=e, commit=c, tail_ms=600) for lo, g, e, c in
        itertools.product((0.05, 0.1, 0.15), (1, 2), (1.0, 1.25, 1.5, 2.0), (0.0, 1.5, 2.0, 2.5, 3.0, 4.0))]
KEYS = ("stride", "lo", "gap", "emin", "commit", "tail_ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--base", nargs="*", default=[], help="checkpoints of the shipped model: scored with the 15 Nov boundary spotter")
    ap.add_argument("--onnx", help="score this exported graph instead of the checkpoints (INT8, as on the phone)")
    ap.add_argument("--sets", help="with --onnx: sets.json of the vocabulary (same-sign set per label)")
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--streams", type=int, default=60, help="streams per kind in each half of the test clips")
    ap.add_argument("--words", type=int, default=6)
    ap.add_argument("--window", type=float, default=1.2)
    ap.add_argument("--no-rule", action="store_true")
    ap.add_argument("--fixed", help="decoder settings to score without tuning, e.g. stride=2,lo=0.1,gap=1,emin=2.5,commit=0,tail_ms=600")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))
    labels = [e["gloss"].upper() for e in entries[:a.subset]]
    if a.onnx:
        model = OnnxWindowModel(a.onnx)
        set_of = np.array(json.loads(pathlib.Path(a.sets).read_text())) if a.sets else None
    else:
        model = WindowModel(a.ckpts, entries, a.subset, rule=not a.no_rule)
        set_of = model.set_of.cpu().numpy() if model.set_of is not None else None
    global TWIN
    if set_of is not None:
        TWIN = {labels[k]: int(set_of[k]) for k in range(len(labels))}
    folds = [make_streams(entries, a.kp_dir, "test", a.subset, a.streams, a.words, a.seed + f, keep=lambda v, f=f: half(v) == f)
             for f in (0, 1)]
    kinds = list(folds[0])
    res = dict(ckpts=a.ckpts, onnx=a.onnx, subset=a.subset, window_s=a.window, streams_per_kind=2 * a.streams, words=a.words, seed=a.seed)

    if a.base:
        rule = None
        if not a.no_rule:
            rule = (signsets.same_sign_sets(entries, a.subset)[0].cuda(), signsets.wlasl_counts(entries, a.subset).cuda())
        clf = le.Classifier(a.base, rule=rule)
        base = [{k: tally([le.run_stream_seg(s[0], s[4], s[3], clf, labels) for s in fold[k]], fold[k]) for k in kinds} for fold in folds]
        res["boundary_15nov"] = {k: report(base[0][k], base[1][k]) for k in kinds}
        res["boundary_15nov_per_stream"] = {k: base[0][k]["per"] + base[1][k]["per"] for k in kinds}
        print("boundary", json.dumps(res["boundary_15nov"]), flush=True)

    post = [{k: [posteriors(s[0], s[4], s[3], model, a.window) for s in fold[k]] for k in kinds} for fold in folds]
    print("windows classified", flush=True)

    def score(f, **kw):
        return {k: tally([run_cont(len(s[0]), s[3], p, hs, labels, set_of, **kw) for s, (p, hs) in zip(folds[f][k], post[f][k])], folds[f][k])
                for k in kinds}

    def mean_wer(*ts):   # mean over the kinds of the pooled word error rate
        return float(np.mean([100 * sum(t[k]["errs"] for t in ts) / sum(t[k]["words"] for t in ts) for k in kinds]))

    def delay(*ts):
        lag = [v for t in ts for k in kinds for v in t[k]["lag"]]
        return float(np.median(lag)) if lag else 1e9

    def pick(rows):      # rows: [(setting, tally, ...)]
        best = min(mean_wer(*r[1:]) for r in rows)
        near = [r for r in rows if mean_wer(*r[1:]) <= best + 1.0]
        return min(near, key=lambda r: delay(*r[1:]))

    if a.fixed:
        best = {}
        for kv in a.fixed.split(","):
            k, v = kv.split("=")
            best[k] = float(v) if k in ("lo", "emin", "commit") else int(v)
        t = [score(0, **best), score(1, **best)]
        res.update(settings=best, continuous={k: report(t[0][k], t[1][k]) for k in kinds},
                   continuous_per_stream={k: t[0][k]["per"] + t[1][k]["per"] for k in kinds})
    else:
        scored = [(g, score(0, **g), score(1, **g)) for g in GRID]
        pick0 = pick([(g, t0) for g, t0, _ in scored])[0]      # best on half 0, scored on half 1
        pick1 = pick([(g, t1) for g, _, t1 in scored])[0]      # best on half 1, scored on half 0
        on1 = next(t1 for g, _, t1 in scored if g is pick0)
        on0 = next(t0 for g, t0, _ in scored if g is pick1)
        final = pick(scored)
        ranked = sorted(scored, key=lambda r: mean_wer(*r[1:]))
        res.update(continuous={k: report(on0[k], on1[k]) for k in kinds}, cross_fit_settings=[pick0, pick1],
                   continuous_per_stream={k: on0[k]["per"] + on1[k]["per"] for k in kinds},
                   settings=final[0], settings_score_in_sample={k: report(final[1][k], final[2][k]) for k in kinds},
                   top=[dict(mean_wer=round(mean_wer(*r[1:]), 1), delay_ms=round(delay(*r[1:])), **r[0]) for r in ranked[:8]],
                   best_by={name: {str(v): round(min(mean_wer(*r[1:]) for r in scored if r[0][name] == v), 1)
                                   for v in sorted({r[0][name] for r in scored}, key=str)}
                            for name in KEYS})
    if a.base:   # paired gain over the boundary spotter on the same streams, bootstrap over streams
        rng = np.random.default_rng(0)
        res["wer_gain_points"] = {}
        for k in kinds:
            d = np.array(res["boundary_15nov_per_stream"][k], float) - np.array(res["continuous_per_stream"][k], float)
            boots = [100 * d[rng.integers(0, len(d), len(d))].sum() / (a.words * len(d)) for _ in range(4000)]
            res["wer_gain_points"][k] = dict(gain=round(100 * float(d.sum()) / (a.words * len(d)), 1),
                                             ci95=[round(float(x), 1) for x in np.percentile(boots, [2.5, 97.5])])
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("per_stream")}, indent=1), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
