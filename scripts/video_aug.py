"""Video-level degradations matching the Track 2 brief: low light, angled signers, moving heads, occlusion,
plus low resolution and compression. Applied to frames *before* MediaPipe, so training keypoints carry the
extractor's real failures (lost hands, jitter) rather than a simulation of them.

A spec is a dict {condition: strength in [0, 1]}; apply() degrades a clip deterministically from a seed.
Training draws random specs (random_spec); evaluation uses the fixed EVAL_SPECS with seeds no training copy uses.
"""
import cv2
import numpy as np

CONDITIONS = ("low_light", "angled", "moving_head", "occlusion", "low_res")

EVAL_SPECS = {
    "clean": {},
    "low_light": {"low_light": 0.7},
    "angled": {"angled": 0.7},
    "moving_head": {"moving_head": 0.7},
    "occlusion": {"occlusion": 0.7},
    "low_res": {"low_res": 0.7},
    "combined": {"low_light": 0.5, "moving_head": 0.5, "occlusion": 0.4},
}


def random_spec(rng):
    """One to three conditions at random strengths; each training copy gets its own mix."""
    k = int(rng.choice([1, 1, 2, 2, 3]))
    picked = rng.choice(len(CONDITIONS), size=k, replace=False)
    return {CONDITIONS[i]: float(rng.uniform(0.3, 1.0)) for i in picked}


def _low_light(frames, s, rng):
    gain = 1.0 - 0.8 * s * rng.uniform(0.8, 1.0)          # down to about 0.2 of the light
    gamma = 1.0 + 1.0 * s * rng.uniform(0.6, 1.0)
    sigma = 2.0 + 12.0 * s * rng.uniform(0.5, 1.0)        # sensor noise after the darkening
    tint = 1.0 + rng.uniform(-0.15, 0.15, size=3) * s     # cheap sensors shift colour in the dark
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0 * gain).astype(np.float32)
    out = []
    for f in frames:
        x = lut[f] * tint
        x = x + rng.normal(0, sigma, size=x.shape).astype(np.float32)
        out.append(np.clip(x, 0, 255).astype(np.uint8))
    return out


def _angled(frames, s, rng):
    """Perspective warp of the signer plane turned by up to about 45 degrees about the vertical axis."""
    h, w = frames[0].shape[:2]
    yaw = np.deg2rad(45 * s * rng.uniform(0.6, 1.0)) * rng.choice([-1, 1])
    squeeze = (1 - np.cos(yaw)) * w / 2
    near = 0.5 * abs(np.sin(yaw)) * h * 0.25
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if yaw > 0:   # right edge turns away: it shrinks vertically
        dst = np.float32([[squeeze, -near], [w - squeeze, near], [w - squeeze, h - near], [squeeze, h + near]])
    else:
        dst = np.float32([[squeeze, near], [w - squeeze, -near], [w - squeeze, h + near], [squeeze, h - near]])
    m = cv2.getPerspectiveTransform(src, dst)
    return [cv2.warpPerspective(f, m, (w, h), borderMode=cv2.BORDER_REPLICATE) for f in frames]


def _moving_head(frames, s, rng):
    """A head-worn camera: smooth drift, rotation and zoom over time, with motion blur from the frame-to-frame motion."""
    h, w = frames[0].shape[:2]
    n = len(frames)
    t = np.arange(n) / 25.0

    def wave(amp):
        f1, f2 = rng.uniform(0.3, 1.2), rng.uniform(1.5, 3.5)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        walk = np.cumsum(rng.normal(0, 1, n))
        walk = (walk - walk.mean()) / (np.abs(walk).max() + 1e-6)
        return amp * (0.5 * np.sin(2 * np.pi * f1 * t + p1) + 0.2 * np.sin(2 * np.pi * f2 * t + p2) + 0.3 * walk)

    tx, ty = wave(0.12 * s * w), wave(0.08 * s * h)
    rot = wave(8.0 * s)
    zoom = 1.0 + wave(0.12 * s)
    out = []
    for i, f in enumerate(frames):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rot[i], zoom[i])
        m[:, 2] += (tx[i], ty[i])
        g = cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if i > 0:
            dx, dy = tx[i] - tx[i - 1], ty[i] - ty[i - 1]
            length = int(min(np.hypot(dx, dy) * 1.5, 15))
            if length >= 2:
                k = np.zeros((length, length), np.float32)
                ang = np.arctan2(dy, dx)
                c = (length - 1) / 2
                for j in range(length):
                    x = int(round(c + (j - c) * np.cos(ang)))
                    y = int(round(c + (j - c) * np.sin(ang)))
                    k[y, x] = 1
                g = cv2.filter2D(g, -1, k / k.sum())
        out.append(g)
    return out


def _occlusion(frames, s, rng):
    """An object in front of the signer for part of the clip, sometimes sliding across; biased to the signing space."""
    h, w = frames[0].shape[:2]
    n = len(frames)
    bw, bh = int(w * rng.uniform(0.15, 0.25 + 0.2 * s)), int(h * rng.uniform(0.15, 0.25 + 0.2 * s))
    x0, y0 = rng.uniform(0.1, 0.9) * w - bw / 2, rng.uniform(0.35, 0.9) * h - bh / 2
    move = rng.random() < 0.4
    x1, y1 = (x0 + rng.uniform(-0.4, 0.4) * w, y0 + rng.uniform(-0.2, 0.2) * h) if move else (x0, y0)
    span = max(int(n * rng.uniform(0.3 + 0.5 * s, 1.0)), 1)
    a = int(rng.integers(0, n - span + 1))
    colour = rng.integers(0, 256, size=3)
    texture = (rng.normal(0, 25, size=(bh, bw, 3)) + colour).clip(0, 255).astype(np.uint8)
    out = []
    for i, f in enumerate(frames):
        if a <= i < a + span:
            u = (i - a) / max(span - 1, 1)
            cx, cy = int(x0 + (x1 - x0) * u), int(y0 + (y1 - y0) * u)
            g = f.copy()
            xa, ya, xb, yb = max(cx, 0), max(cy, 0), min(cx + bw, w), min(cy + bh, h)
            if xb > xa and yb > ya:
                g[ya:yb, xa:xb] = texture[ya - cy:yb - cy, xa - cx:xb - cx]
            out.append(g)
        else:
            out.append(f)
    return out


def _low_res(frames, s, rng):
    h, w = frames[0].shape[:2]
    scale = 1.0 - 0.75 * s * rng.uniform(0.7, 1.0)
    q = int(90 - 70 * s * rng.uniform(0.6, 1.0))
    out = []
    for f in frames:
        g = cv2.resize(f, (max(int(w * scale), 16), max(int(h * scale), 16)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        g = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def enhance(frames, dark=0.35, target=0.45):
    """Run-time low-light fix, applied before the landmarker on the device too: when a clip's mean
    brightness is below `dark`, lift it with a gamma that maps the mean to `target`, then equalise local
    contrast on the lightness channel. Clips of normal brightness are returned untouched."""
    if not frames:
        return frames
    mean = float(np.mean([f[::4, ::4].mean() for f in frames[::max(len(frames) // 8, 1)]])) / 255.0
    if mean >= dark:
        return frames
    gamma = np.log(target) / np.log(max(mean, 1e-3))
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = []
    for f in frames:
        g = cv2.cvtColor(lut[f], cv2.COLOR_RGB2LAB)
        g[..., 0] = clahe.apply(g[..., 0])
        out.append(cv2.cvtColor(g, cv2.COLOR_LAB2RGB))
    return out


_ORDER = [("angled", _angled), ("moving_head", _moving_head), ("occlusion", _occlusion),
          ("low_light", _low_light), ("low_res", _low_res)]


def apply(frames, spec, seed):
    """Degrade a list of RGB uint8 frames. Geometry first, then the object, then the sensor."""
    rng = np.random.default_rng(seed)
    for name, fn in _ORDER:
        if spec.get(name, 0) > 0:
            frames = fn(frames, float(spec[name]), rng)
    return frames
