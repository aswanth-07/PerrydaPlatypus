"""Continuous signing: signs made one after another with the hands kept up, built from isolated clips.

An isolated clip is rest -> the hands rise -> the sign -> the hands drop -> rest. A person who signs several
words in a row does not go back to rest: the hands move straight from the end of one sign to the start of the
next, sometimes with a pause in the air. No gloss-annotated continuous ASL is on disk, so that is made here from
the isolated clips, in two places:

  cont_streams   raw keypoint streams for measuring live captions (scripts/live_cont.py): the core of each clip
                 (the part between the rise and the drop) joined to the next by an interpolated movement, with
                 pauses in the air for some joins and real rests for others. Every clip is first moved into one
                 common body frame, so the stream looks like one signer standing still.
  windows        training windows for the live model (train.py --cont): a window of the width the phone
                 classifies, centred somewhere in such a stream. Centred on a sign, the label is that sign
                 whatever else the window holds; centred between two signs, the label is "no sign".

The second follows the online recogniser of Zuo, Wei and Mak (EMNLP 2024): an isolated-sign model slid over a
stream fails unless it was trained on windows as it will meet them, with the movements between signs as a
background class. Its window answers are then turned into one caption per sign (`Decoder` below; Spotter.kt
is the same logic).

The core of a clip is found from the wrists' height alone (`core_span`): the stretch above the rest line, less
the rise at its start and the drop at its end.
"""
import math

import numpy as np
import torch

import normalise as dv

REST_H = -1.0        # rest line: the higher wrist, in shoulder widths above the shoulder line (as Spotter.kt)
CORE_FRAC = 0.65     # the core starts once the wrist has covered this share of its rise (and ends likewise)
ANCHOR_W = 0.22      # shoulder width of the common body frame, in a unit square
ANCHOR_C = np.array([0.5, 0.4, 0.0])
ARM_POSE = [dv.POSE_KEEP.index(k) for k in (13, 14, 15, 16)]   # elbows and wrists
ARMS = ARM_POSE + list(range(dv.LH[0], dv.RH[1]))              # every point that belongs to the arms and hands


def core_span(h, rest_h=REST_H, frac=CORE_FRAC, min_len=5):
    """h: the higher wrist's height per frame (NaN: no body). Returns (s0, a, b, s1): the stretch above the rest
    line is [s0, s1) and the sign's core [a, b), both as frame indices."""
    t = len(h)
    hh = np.where(np.isfinite(h), h, -9.0)
    up = hh > rest_h
    if not up.any():
        return 0, 0, t, t
    s0, s1 = int(np.argmax(up)), t - int(np.argmax(up[::-1]))
    seg = hh[s0:s1]
    peak = float(np.percentile(seg, 90))
    lo0, lo1 = min(float(seg[0]), peak), min(float(seg[-1]), peak)
    a = s0 + int(np.argmax(seg >= lo0 + frac * (peak - lo0)))
    b = s1 - int(np.argmax(seg[::-1] >= lo1 + frac * (peak - lo1)))
    if b - a < max(min_len, 0.35 * (s1 - s0)):
        a, b = s0, s1
    return s0, a, b, s1


# ---------------------------------------------------------------------------------------------------------
# Raw streams for measurement
# ---------------------------------------------------------------------------------------------------------

def anchor(raw, wh):
    """One clip's raw arrays moved into the common body frame (a unit square: mean shoulder midpoint at
    ANCHOR_C, median shoulder width ANCHOR_W). None when the shoulders were never seen."""
    w, h = float(wh[0]), float(wh[1])
    side = max(w, h)
    asp = np.array([w / side, h / side, w / side])
    xyz = raw["pose"][..., :3] * asp
    a, b = xyz[:, 11], xyz[:, 12]
    ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    if not ok.any():
        return None
    centre = ((a[ok] + b[ok]) / 2).mean(0)
    scale = max(float(np.median(np.linalg.norm((a[ok] - b[ok])[:, :2], axis=1))), 1e-3)

    def f(p):
        return (p * asp - centre) / scale * ANCHOR_W + ANCHOR_C

    pose = raw["pose"].copy()
    pose[..., :3] = f(pose[..., :3])
    return dict(pose=pose, hands=f(raw["hands"]), face=f(raw["face"]), hands_world=raw["hands_world"].copy())


def heights(raw):
    """The higher wrist's height above the shoulder line in shoulder widths, per frame of an anchored clip."""
    p = raw["pose"]
    mid = (p[:, 11, 1] + p[:, 12, 1]) / 2
    sw = np.nanmedian(np.hypot(p[:, 11, 0] - p[:, 12, 0], p[:, 11, 1] - p[:, 12, 1]))
    hs = np.stack([(mid - p[:, k, 1]) / max(sw, 1e-3) for k in (15, 16)], 1)
    return np.where(np.isfinite(hs).any(1), np.nanmax(np.where(np.isfinite(hs), hs, -np.inf), 1), np.nan)


GROUPS = (("pose", None), ("hands", 0), ("hands", 1), ("face", None), ("hands_world", 0), ("hands_world", 1))


def _get(frame, key, slot):
    return frame[key] if slot is None else frame[key][slot]


def _endpoint(frames, last):
    """The pose a join starts from (last=True: the end of a core) or arrives at: per group, the nearest seen
    value within five frames of that end, so one dropped hand detection does not make the hand vanish."""
    order = frames[::-1][:5] if last else frames[:5]
    out = {k: np.full_like(frames[0][k], np.nan) for k in frames[0]}
    for key, slot in GROUPS:
        for f in order:
            v = _get(f, key, slot)
            if np.isfinite(v.flat[0]):
                if slot is None:
                    out[key] = v.copy()
                else:
                    out[key][slot] = v
                break
    return out


def _blend(p, q, w):
    """Between two poses, group by group: interpolated when both ends have the group, otherwise the nearer end's."""
    out = {k: np.full_like(p[k], np.nan) for k in p}
    for key, slot in GROUPS:
        a, b = _get(p, key, slot), _get(q, key, slot)
        fa, fb = np.isfinite(a.flat[0]), np.isfinite(b.flat[0])
        v = a * (1 - w) + b * w if fa and fb else (a if w < 0.5 else b)
        if slot is None:
            out[key] = v
        else:
            out[key][slot] = v
    return out


def _jitter(frame, rng, sigma=0.0015):
    """A held pose is never perfectly still to a tracker."""
    return {k: (v + rng.normal(0, sigma, v.shape) if k != "hands_world" else v + rng.normal(0, sigma / 4, v.shape))
            for k, v in frame.items()}


def join_frames(tail, head, fps, rng, kind, hover=None):
    """The frames between the end of one core and the start of the next. kind: "direct" (one movement),
    "pause" (a hold in the air, 0.3-1.5 s, before or after the movement), "hover" (the hands go to the relaxed
    pose `hover`, wait there 0.5-1.5 s, then move on)."""
    p, q = _endpoint(tail, True), _endpoint(head, False)

    def move(a, b):
        n = max(int(round(rng.uniform(0.2, 0.5) * fps)), 3)
        ws = [k / (n + 1) for k in range(1, n + 1)]
        return [_jitter(_blend(a, b, w * w * (3 - 2 * w)), rng, 0.0008) for w in ws]

    if kind == "hover" and hover is not None:
        hold = int(round(rng.uniform(0.5, 1.5) * fps))
        return move(p, hover) + [_jitter(hover, rng) for _ in range(hold)] + move(hover, q)
    mv = move(p, q)
    if kind != "pause":
        return mv
    hold = int(round(rng.uniform(0.3, 1.5) * fps))
    if rng.random() < 0.6:      # the last pose is held, then the hands move on
        return [_jitter(p, rng) for _ in range(hold)] + mv
    return mv + [_jitter(q, rng) for _ in range(hold)]


JOINS = ("direct", "pause", "rest", "hover")


def cont_streams(entries, kp_dir, split, subset, n_streams, n_words, seed, mix=(0.5, 0.3, 0.0, 0.2), keep=None):
    """Streams of n_words signs. mix: the share of joins that are (direct, pause, rest, hover); a rest join keeps
    both clips' own drop and rise; a hover join waits at a relaxed pose in the air, taken from half-way through
    the previous clip's drop or the next clip's rise (a pause when neither clip has one). The first sign keeps its
    own rise and the last its own drop. keep: a predicate on the video id, to draw from part of the split only.
    Returns a list of (frames, truth, bounds, fps, wh, kinds): bounds are the cores' frame ranges."""
    import pathlib
    rng = np.random.default_rng(seed)
    kp = pathlib.Path(kp_dir)
    pool = [(i["video_id"], e["gloss"].upper()) for e in entries[:subset] for i in e["instances"]
            if i["split"] == split and (kp / f"{i['video_id']}.npz").exists() and (keep is None or keep(i["video_id"]))]
    out = []
    while len(out) < n_streams:
        pick = [pool[j] for j in rng.choice(len(pool), n_words, replace=False)]
        clips = []
        for vid, g in pick:
            d = np.load(kp / f"{vid}.npz")
            raw = {k: d[k].astype(np.float64) for k in ("pose", "hands", "face", "hands_world")}
            a = anchor(raw, d["wh"])
            if a is None:
                break
            fr = [{k: a[k][t] for k in a} for t in range(len(a["pose"]))]
            clips.append((fr, core_span(heights(a)), float(d["fps"]), g))
        if len(clips) < n_words:
            continue
        fps = clips[0][2]
        frames, truth, bounds, kinds = [], [], [], []
        for n, (fr, (s0, a, b, _s1), _, g) in enumerate(clips):
            if n == 0:
                frames += fr[:a]
            else:
                kind = JOINS[int(rng.choice(len(JOINS), p=mix))]
                prev_fr, (_, pa, pb, ps1), _, _ = clips[n - 1]
                hover = None
                if kind == "hover":
                    poses = ([prev_fr[(pb + ps1) // 2]] if ps1 - pb >= 3 else []) + ([fr[(s0 + a) // 2]] if a - s0 >= 3 else [])
                    if poses:
                        hover = poses[int(rng.integers(len(poses)))]
                    else:
                        kind = "pause"
                kinds.append(kind)
                if kind == "rest":
                    frames += prev_fr[pb:] + fr[:a]
                else:
                    frames += join_frames(prev_fr[pa:pb], fr[a:b], fps, rng, kind, hover)
            start = len(frames)
            frames += fr[a:b]
            truth.append(g)
            bounds.append((start, len(frames)))
        frames += clips[-1][0][clips[-1][1][2]:]
        out.append((frames, truth, bounds, fps, np.array([1, 1]), kinds))
    return out


# ---------------------------------------------------------------------------------------------------------
# Training windows
# ---------------------------------------------------------------------------------------------------------

def clip_spans(X, L, rest_h=REST_H, frac=CORE_FRAC):
    """Core of every cached (normalised) clip. X: N x T x P x 3, L: N. Returns an N x 4 LongTensor (s0, a, b, s1)."""
    wr = [dv.POSE_KEEP.index(15), dv.POSE_KEEP.index(16)]
    y = X[:, :, wr + list(dv.SHOULDERS), 1].float().cpu().numpy()
    Ln = L.cpu().numpy()
    out = np.zeros((len(Ln), 4), np.int64)
    for n in range(len(Ln)):
        t = int(Ln[n])
        if t < 2:
            out[n] = (0, 0, max(t, 1), max(t, 1))
            continue
        mid = (y[n, :t, 2] + y[n, :t, 3]) / 2
        h = np.maximum(mid - y[n, :t, 0], mid - y[n, :t, 1])   # normalised points are already in shoulder widths
        out[n] = core_span(h, rest_h, frac)
    return torch.from_numpy(out)


def _fetch(Z, ids, pos):
    """Z: N x T x ..., ids: B, pos: B x t fractional frame positions -> B x t x ... linearly interpolated."""
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=Z.shape[1] - 1)
    lo = lo.clamp(0, Z.shape[1] - 1)
    a = (pos - pos.floor()).view(*pos.shape, *([1] * (Z.dim() - 2)))
    r = ids[:, None]
    z0, z1 = Z[r, lo].float(), Z[r, hi].float()
    return z0 + (z1 - z0) * a


def _fill(v, has):
    """v: B x t x ..., has: B x t bools. Steps without a value take the linear interpolation between the nearest
    steps with one (the nearest alone at the ends); rows with no value at all become zero."""
    b, t = has.shape
    idx = torch.arange(t, device=v.device)[None].expand(b, t)
    prev = torch.cummax(torch.where(has, idx, torch.full_like(idx, -1)), 1).values
    nxt = torch.flip(torch.cummin(torch.flip(torch.where(has, idx, torch.full_like(idx, t)), [1]), 1).values, [1])
    p = torch.where(prev >= 0, prev, nxt).clamp(0, t - 1)
    n = torch.where(nxt < t, nxt, prev).clamp(0, t - 1)
    wgt = torch.where(n != p, (idx - p).float() / (n - p).clamp_min(1).float(), torch.zeros(b, t, device=v.device))
    wgt = wgt.view(b, t, *([1] * (v.dim() - 2)))
    r = torch.arange(b, device=v.device)[:, None]
    out = v[r, p] * (1 - wgt) + v[r, n] * wgt
    return out * has.any(1).view(b, *([1] * (v.dim() - 1)))


def windows(data, spans, left, perm, c, pool, t_in, flip_fn, none_share=0.25, rest_p=0.25, pause_p=0.3, hover_p=0.15):
    """One training window per clip of the batch.

    data: X, W, S, L on the GPU; spans: clip_spans(); left: per clip, whether the left hand is the dominant one
    (the model was trained with every clip flipped so that the right hand is); c: the batch's clip rows;
    pool: rows neighbours are drawn from; flip_fn: train.apply_flip.
    Returns x, w, s (B x t_in ...) and is_none (B bools): the window is centred between two signs.

    Time is measured in the centre clip's frames and every length is a share of the window, which on the phone is
    a fixed 1.2 s: the core is 0.42-1.8 windows long (0.5-2.2 s), a movement between two signs 0.1-0.35 windows
    (0.12-0.42 s), a pause in the air 0.15-1.6 windows (longer than a window, so some windows hold nothing but a
    held pose). Each side of the sign is another sign (after such a join), the clip's own rise or drop from rest
    (rest_p), or a hover (hover_p): the hands wait, still, at a relaxed pose taken from half-way through the
    neighbour's rise or drop. The arms and hands come from the stream; head, face, shoulders and hips stay the
    centre clip's own throughout, so nothing but the hands tells the model where a sign ends.
    """
    dev = c.device
    b = len(c)
    X, W, S, L = data["X"], data["W"], data["S"], data["L"]
    def u01(lo=0.0, hi=1.0):
        return torch.empty(b, 1, device=dev).uniform_(lo, hi)
    ia = pool[torch.randint(len(pool), (b,), device=dev)]
    ic = pool[torch.randint(len(pool), (b,), device=dev)]
    sp, spa, spc = spans[c].float(), spans[ia].float(), spans[ic].float()
    a0, b0 = sp[:, 1:2], sp[:, 2:3]
    d = (b0 - a0).clamp_min(2.0)
    width = d * torch.exp(u01(math.log(0.55), math.log(2.4)))
    is_none = torch.rand(b, device=dev) < none_share
    rest_l, rest_r = torch.rand(b, 1, device=dev) < rest_p, torch.rand(b, 1, device=dev) < rest_p
    rest_r = rest_r & ~(is_none[:, None] & (torch.rand(b, 1, device=dev) < 0.8))
    hov_l = (torch.rand(b, 1, device=dev) < hover_p) & ~rest_l
    hov_r = (torch.rand(b, 1, device=dev) < hover_p * (1 + is_none[:, None].float())) & ~rest_r   # twice as often under "no sign"

    def joint():
        move = width * u01(0.10, 0.35)
        pause = torch.where(torch.rand(b, 1, device=dev) < pause_p, width * u01(0.15, 1.6), torch.zeros(b, 1, device=dev))
        first = torch.rand(b, 1, device=dev) < 0.6      # the pause comes before the movement (else after it)
        total = move + pause
        return total, torch.where(first, pause, torch.zeros_like(pause)) / total, torch.where(first, torch.zeros_like(pause), pause) / total

    tl, hl0, hl1 = joint()
    tr, hr0, hr1 = joint()
    # Where the window is centred: anywhere in the core for a sign; for "no sign", inside the join to the next
    # sign (its middle 70%), anywhere from the join into the wait when that side is a hover, or past the core in
    # the clip's own drop when it is a rest.
    drop = (L[c][:, None].float() - b0).clamp_min(0) + 0.3 * width
    centre = torch.where(is_none[:, None],
                         b0 + torch.where(rest_r, u01(0.3, 1.0) * drop,
                                          torch.where(hov_r, 0.15 * tr + u01() * (0.85 * tr + width), (0.15 + 0.7 * u01()) * tr)),
                         a0 + u01() * d)
    s = centre + (torch.linspace(-0.5, 0.5, t_in, device=dev)[None]) * width          # B x t stream time

    def ease(v):
        return v * v * (3 - 2 * v)
    Lb, La, Lc = L[c][:, None].float() - 1, L[ia][:, None].float() - 1, L[ic][:, None].float() - 1
    last_b = b0 - 1
    # A neighbour's pose at the join: the start (end) of its core, or for a hover a still pose half-way through
    # its rise (drop), held for as long as the window reaches.
    first_c = torch.where(hov_r, (spc[:, 0:1] + spc[:, 1:2]) / 2, spc[:, 1:2])
    last_a = torch.where(hov_l, torch.minimum((spa[:, 2:3] + spa[:, 3:4]) / 2, La), spa[:, 2:3] - 1)
    rate_a = torch.where(hov_l, torch.zeros(b, 1, device=dev), (spa[:, 2:3] - spa[:, 1:2]).clamp_min(2) / (width * u01(0.45, 1.6)))
    rate_c = torch.where(hov_r, torch.zeros(b, 1, device=dev), (spc[:, 2:3] - spc[:, 1:2]).clamp_min(2) / (width * u01(0.45, 1.6)))

    in_core = (s >= a0) & (s <= last_b)
    right, lefts = s > last_b, s < a0
    # Right of the core.
    ph = ((s - last_b) / tr).clamp(0, 1)
    wr_c = ease(((ph - hr0) / (1 - hr0 - hr1).clamp_min(1e-3)).clamp(0, 1))
    beyond_r = s > last_b + tr
    pos_c = torch.where(beyond_r, first_c + (s - last_b - tr) * rate_c, first_c.expand_as(s))
    pos_c = torch.minimum(pos_c.clamp_min(0), Lc)
    w_c = torch.where(right & ~rest_r, torch.where(beyond_r, torch.ones_like(s), wr_c), torch.zeros_like(s))
    # Left of the core.
    ph = ((a0 - s) / tl).clamp(0, 1)
    wl_a = ease(((ph - hl1) / (1 - hl0 - hl1).clamp_min(1e-3)).clamp(0, 1))
    beyond_l = s < a0 - tl
    pos_a = torch.where(beyond_l, last_a - (a0 - tl - s) * rate_a, last_a.expand_as(s))
    pos_a = torch.minimum(pos_a.clamp_min(0), La)
    w_a = torch.where(lefts & ~rest_l, torch.where(beyond_l, torch.ones_like(s), wl_a), torch.zeros_like(s))
    w_b = 1 - w_a - w_c
    # The centre clip's own frames: its core, and its own rise or drop on a rest side.
    own = torch.minimum(s.clamp_min(0), Lb)
    pos_b = torch.where(in_core | (right & rest_r) | (lefts & rest_l), own,
                        torch.where(right, last_b.expand_as(s), a0.expand_as(s)))
    # Head, face and trunk: the centre clip's, reflected past its ends so it keeps moving.
    span = Lb.clamp_min(1)
    body = (s % (2 * span))
    body = torch.where(body > span, 2 * span - body, body)

    def source(ids, pos):
        x, w, sn = _fetch(X, ids, pos), _fetch(W, ids, pos), _fetch(S, ids, pos)
        return flip_fn(x, w, sn, left[ids], perm)

    xa, wa, sa = source(ia, pos_a)
    xb, wb, sb = source(c, pos_b)
    xc, wc, sc = source(ic, pos_c)
    x, w, sn = source(c, body)
    x, w, sn = x.clone(), w.clone(), sn.clone()

    def mix(parts, weights, present):
        """Weighted mean over the sources that have the group at all. Steps where none of the sources in play
        has it (a hand the centre sign never shows) are filled from the nearest steps that do, the way
        normalise.fill_group fills a hand the tracker lost; all zero when no source has it."""
        num, den = 0, 0
        for v, wt, pr in zip(parts, weights, present):
            k = (wt * pr).view(b, t_in, *([1] * (v.dim() - 2)))
            num, den = num + v * k, den + k
        return _fill(num / den.clamp_min(1e-6), den.reshape(b, t_in) > 0)

    ws = (w_a, w_b, w_c)
    one = torch.ones(b, 1, device=dev)
    x[:, :, ARM_POSE] = mix([v[:, :, ARM_POSE] for v in (xa, xb, xc)], ws, (one, one, one))
    for slot, (lo, hi) in ((1, dv.LH), (2, dv.RH)):
        pres = [(S[i][:, :, slot].float().sum(1, keepdim=True) > 0).float() for i in (ia, c, ic)]
        x[:, :, lo:hi] = mix([v[:, :, lo:hi] for v in (xa, xb, xc)], ws, pres)
        w[:, :, slot - 1] = mix([v[:, :, slot - 1] for v in (wa, wb, wc)], ws, pres)
        sn[:, :, slot] = w_a * sa[:, :, slot] * pres[0] + w_b * sb[:, :, slot] * pres[1] + w_c * sc[:, :, slot] * pres[2]
    return x, w, sn, is_none


# ---------------------------------------------------------------------------------------------------------
# The decoder: window answers -> captions. Spotter.kt (continuous mode) is the same logic; keep them identical.
# ---------------------------------------------------------------------------------------------------------

TOPK = 5


def top_answers(probs, mass, k=TOPK):
    """A window's answer in the form the decoder takes: its k most probable words (index, probability, the
    probability of the word's whole same-sign set), most probable first, ties to the lower index."""
    order = np.argsort(-np.asarray(probs, np.float32), kind="stable")[:k]
    return [int(i) for i in order], [float(probs[i]) for i in order], [float(mass[i]) for i in order]


class Decoder:
    """One caption per sign, committed when the sign ends, the way live speech captions settle a word.

    A window is a sign window when its top word's set reaches `lo` and beats "no sign"; anything else (the
    movement between two signs, a window without hands) is a gap window. A segment is a run of sign windows; it
    closes after `gap` gap windows in a row, or when the hands come to rest (`flush`). While it is open every
    window adds evidence: each of its top words' sets gains that set's probability. When it closes, the set with
    the most evidence is the sign, if its evidence reaches `emin` (a sign seen in too few windows, or never with
    any confidence, is dropped); the word shown is that set's word with the most summed probability.

    Committing at the end is what removes the false words of a decoder that answers as soon as a few windows
    agree: on the way into a sign the window holds half of it, and half a sign often looks like another sign.
    A sign held for long is not kept waiting: once the leading set's evidence reaches `commit` it is captioned at
    once, and the segment's close then adds nothing. `tentative` is the leading word of the open segment, for the
    display to show, unsettled, before the caption commits.

    Voting over the last few windows (Zuo, Wei and Mak, EMNLP 2024, online post-processing) was measured first
    and is what this replaces: see README, "Live captions".
    """

    def __init__(self, set_of=None, lo=0.1, gap=1, emin=2.5, commit=0.0, show=1.0):
        self.set_of, self.lo, self.gap, self.emin, self.commit, self.show = set_of, lo, gap, emin, commit, show
        self.reset()

    def reset(self):
        self.ev, self.wsum, self.gap_run, self.done = {}, {}, 0, None

    def _set(self, k):
        return k if self.set_of is None else int(self.set_of[k])

    def _leader(self):
        s = max(self.ev, key=self.ev.get)          # ties: the set that entered the segment first
        w = max((k for k in self.wsum if self._set(k) == s), key=self.wsum.get)
        return s, w, self.ev[s]

    @property
    def tentative(self):
        """The open segment's leading word once its evidence reaches `show` (and before a caption commits), else None."""
        if not self.ev or self.done is not None:
            return None
        s, w, e = self._leader()
        return w if e >= self.show else None

    def push(self, top, none):
        """top: top_answers() of the window, or None when it had no hands; none: its probability of "no sign".
        Returns the word index of a caption that has just been committed, or None."""
        if top is None or top[2][0] < self.lo or top[2][0] <= none:
            self.gap_run += 1
            return self.flush() if self.gap_run >= self.gap else None
        self.gap_run = 0
        seen = set()
        for k, p, m in zip(*top):
            s = self._set(k)
            self.wsum[k] = self.wsum.get(k, 0.0) + p
            if s not in seen:
                self.ev[s] = self.ev.get(s, 0.0) + m
                seen.add(s)
        if self.commit > 0 and self.done is None:
            s, w, e = self._leader()
            if e >= self.commit:
                self.done = s
                return w
        return None

    def flush(self):
        """Close the open segment (a gap, or the hands at rest): its caption, or None."""
        word = None
        if self.ev and self.done is None:
            s, w, e = self._leader()
            if e >= self.emin:
                word = w
        self.reset()
        return word


class RestGate:
    """Whether the classifier should run at this frame: from the frame a wrist rises above the rest line until
    `tail_ms` after both have been back below it for `rest_ms` (the window's centre is still on the last sign)."""

    def __init__(self, rest_h=REST_H, rest_ms=200, tail_ms=600):
        self.rest_h, self.rest_ms, self.tail_ms = rest_h, rest_ms, tail_ms
        self.open, self.low_since, self.closed_at = False, -1, None

    def push(self, ts, h):
        up = h is not None and h > self.rest_h + (0.0 if self.open else 0.1)
        if self.open:
            if up:
                self.low_since = -1
            elif self.low_since < 0:
                self.low_since = ts
            if self.low_since >= 0 and ts - self.low_since >= self.rest_ms:
                self.open, self.closed_at = False, ts
        elif up:
            self.open, self.low_since = True, -1
        return self.open or (self.closed_at is not None and ts - self.closed_at <= self.tail_ms)
