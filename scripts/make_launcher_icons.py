"""The app's launcher icons from the glass-hand artwork (an open hand between sound rings, on pure black).

Foreground (mipmap-*/ic_launcher_foreground.png): the artwork scaled so everything brighter than near-black stays
inside the 66 dp circle every launcher shows of the 108 dp canvas (radius 0.31; the faintest glow may touch it),
with black turned into transparency (alpha = brightest channel, colour divided back out), so it looks the same over
the black background layer. The splash screen shows the same bitmap.

Monochrome (mipmap-*/ic_launcher_monochrome.png, Android 13+ themed icons, tinted by the launcher): a flat white
silhouette found once at the artwork's full resolution and then area-averaged down, so its edges are anti-aliased.
The glass is dark inside (brightest channel ~80) with a bright rim (150-255) that glows outward, so a low threshold
inflates the hand and glues the inner ring to the thumb. Instead: threshold at the rim, close small gaps in it, fill
the interior, then open with a disc wider than a ring and narrower than a finger so the rings fall away; a light
blur-and-threshold rounds off what is left where a ring met the thumb. The rings are drawn as the two outer arcs on
each side (the innermost pair is dim and half hidden by the hand and the thumb), widened to an even stroke with
round ends so they hold up at launcher size, and kept apart from the hand.

Play Store (src/main/ic_launcher-playstore.png): 512 x 512, opaque, full square; Play rounds the corners.

Usage: uv run --with pillow --with numpy --with scipy python scripts/make_launcher_icons.py \
           <artwork> android/app/src/main [--preview preview.png]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DENSITIES = {'mdpi': 108, 'hdpi': 162, 'xhdpi': 216, 'xxhdpi': 324, 'xxxhdpi': 432}
VISIBLE_RADIUS = 0.31       # of the 108 dp canvas: 66 dp safe circle / 2 / 108 = 0.306
GLOW_FLOOR = 40             # brightest channel above this counts as artwork when measuring its extent


def grow(mask, r):
    return ndimage.distance_transform_edt(~mask) <= r


def shrink(mask, r):
    return ndimage.distance_transform_edt(mask) > r


def content_radius(lum):
    """How far from the centre the artwork reaches, as a fraction of its width."""
    h, w = lum.shape
    ys, xs = np.nonzero(lum > GLOW_FLOOR)
    return np.sqrt((xs - w / 2) ** 2 + (ys - h / 2) ** 2).max() / w


def silhouette(lum, hand_t=110, close_r=12, open_r=30, smooth=5.0, ring_t=130, gap=30, stroke=7):
    """Monochrome alpha (0..1) at the artwork's resolution: the solid hand and four arcs."""
    h, w = lum.shape
    seed = (int(0.57 * h), int(0.50 * w))           # the palm
    b = lum > hand_t
    b = shrink(grow(b, close_r), close_r)            # seal gaps in the rim
    b = ndimage.binary_fill_holes(b)
    opened = grow(shrink(b, open_r), open_r)         # the rings and their glow fall away
    lab, _ = ndimage.label(opened)
    assert lab[seed], 'palm seed is not inside the hand'
    hand = lab == lab[seed]
    hand = ndimage.gaussian_filter(hand.astype(np.float32), smooth) > 0.5
    hand = ndimage.binary_fill_holes(hand)

    # Arcs: the bright ring cores clear of the hand; the four largest are the outer two on each side.
    core = (lum > ring_t) & ~grow(hand, gap)
    lab, n = ndimage.label(core)
    sizes = ndimage.sum(core, lab, range(1, n + 1))
    biggest = np.argsort(sizes)[::-1][:4] + 1
    sides = [ndimage.center_of_mass(lab == i)[1] < w / 2 for i in biggest]
    assert sum(sides) == 2, f'expected two arcs on each side, got {sum(sides)} on the left'
    arcs = grow(np.isin(lab, biggest), stroke)
    clearance = ndimage.distance_transform_edt(~hand)[arcs].min()
    print(f'arcs: {sorted(int(s) for s in sizes[biggest - 1])} px, clearance from the hand {clearance:.0f} px')
    return (hand | arcs).astype(np.float32)


def place(img, size, scale, resample):
    """img scaled to scale * size and centred on a transparent (or black, for RGB) size x size canvas."""
    inner = round(size * scale)
    canvas = Image.new(img.mode, (size, size), 0)
    off = (size - inner) // 2
    canvas.paste(img.resize((inner, inner), resample), (off, off))
    return canvas


def to_alpha(rgb):
    """Black becomes transparent: alpha is the brightest channel, colour is divided back out."""
    a = np.asarray(rgb).astype(np.float32)
    alpha = a.max(axis=2)
    col = np.where(alpha[..., None] > 0, a * 255.0 / np.maximum(alpha[..., None], 1), 0)
    return Image.fromarray(np.dstack([np.clip(col, 0, 255), alpha]).astype(np.uint8), 'RGBA')


def white(alpha_l):
    a = np.asarray(alpha_l)
    return Image.fromarray(np.dstack([np.full(a.shape + (3,), 255, np.uint8), a]), 'RGBA')


def preview(art, mono_l, scale, path):
    """Launcher masks (circle, squircle, rounded square), themed icons light and dark, and true-size copies."""
    P = 216
    view = place(art, P * 3 // 2, scale, Image.LANCZOS).crop((P // 4, P // 4, P // 4 + P, P // 4 + P))
    mono = place(mono_l, P * 3 // 2, scale, Image.BOX).crop((P // 4, P // 4, P // 4 + P, P // 4 + P))

    def mask(shape, s):
        big = s * 4
        if shape == 'squircle':                     # |x|^4 + |y|^4 <= 1
            yy, xx = np.mgrid[-1:1:complex(0, big), -1:1:complex(0, big)]
            m = Image.fromarray(((np.abs(xx) ** 4 + np.abs(yy) ** 4) <= 1).astype(np.uint8) * 255)
        else:
            m = Image.new('L', (big, big), 0)
            d = ImageDraw.Draw(m)
            if shape == 'circle':
                d.ellipse((0, 0, big - 1, big - 1), fill=255)
            else:
                d.rounded_rectangle((0, 0, big - 1, big - 1), radius=big // 5, fill=255)
        return m.resize((s, s), Image.LANCZOS)

    def themed(bg, fg, s, src):
        tile = Image.new('RGB', (s, s), bg)
        tile.paste(Image.new('RGB', (s, s), fg), (0, 0), src)
        return tile

    tiles = [(view, mask(s, P)) for s in ('circle', 'squircle', 'rounded')]
    tiles.append((themed((191, 216, 227), (0, 74, 107), P, mono), mask('circle', P)))     # light themed
    tiles.append((themed((0, 50, 73), (191, 216, 227), P, mono), mask('circle', P)))      # dark themed
    sheet = Image.new('RGB', (10 + len(tiles) * (P + 10), P + 20 + 126 + 10), (242, 242, 242))
    for i, (tile, m) in enumerate(tiles):
        sheet.paste(tile, (10 + i * (P + 10), 10), m)
    # True size: a 48 dp icon on a 420 dpi phone is 126 px across; and on a 160 dpi screen, 48 px.
    for j, s in enumerate((126, 48)):
        x = 10 + j * 300
        sheet.paste(view.resize((s, s), Image.LANCZOS), (x, P + 20), mask('circle', s))
        sheet.paste(themed((191, 216, 227), (0, 74, 107), s, mono.resize((s, s), Image.BOX)),
                    (x + s + 10, P + 20), mask('circle', s))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('artwork')
    ap.add_argument('main_dir', help='android/app/src/main')
    ap.add_argument('--preview')
    args = ap.parse_args()

    art = Image.open(args.artwork).convert('RGB')
    assert art.width == art.height, 'the artwork must be square'
    lum = np.asarray(art).astype(np.float32).max(axis=2)
    scale = VISIBLE_RADIUS / content_radius(lum)
    print(f'artwork {art.width} px, reaches {content_radius(lum):.3f} of its width; scaled to {scale:.3f}')
    mono_l = Image.fromarray((silhouette(lum) * 255).astype(np.uint8), 'L')

    for name, size in DENSITIES.items():
        d = os.path.join(args.main_dir, 'res', f'mipmap-{name}')
        os.makedirs(d, exist_ok=True)
        to_alpha(place(art, size, scale, Image.LANCZOS)).save(os.path.join(d, 'ic_launcher_foreground.png'),
                                                              optimize=True)
        white(place(mono_l, size, scale, Image.BOX)).save(os.path.join(d, 'ic_launcher_monochrome.png'),
                                                          optimize=True)

    place(art, 512, 0.9, Image.LANCZOS).save(os.path.join(args.main_dir, 'ic_launcher-playstore.png'),
                                             optimize=True)
    if args.preview:
        preview(art, mono_l, scale, args.preview)


if __name__ == '__main__':
    main()
