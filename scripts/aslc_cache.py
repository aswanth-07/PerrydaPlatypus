"""Cache of extracted ASL Citizen keypoints for classes below --max-cls (WLASL-ordered), thinned to --t-max frames.

Usage: aslc_cache.py <manifest.json> <kp_dir> <out.pt> [--max-cls 2000] [--t-max 48]
       aslc_cache.py ... --check   (print how many in-range clips are extracted, then exit)
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data_v2 as dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("kp_dir")
    ap.add_argument("out")
    ap.add_argument("--max-cls", type=int, default=2000)
    ap.add_argument("--t-max", type=int, default=48)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    m = json.load(open(a.manifest, encoding="utf-8"))
    want = {"aslc_" + pathlib.Path(c["file"]).stem for c in m if c["cls"] < a.max_cls}
    have = {p.stem for p in pathlib.Path(a.kp_dir).glob("aslc_*.npz")} & want
    if a.check:
        print(f"{len(have)} {len(want)}")
        return
    c = dv.build(a.kp_dir, "holistic", have, a.t_max)
    torch.save(c, a.out)
    print(f"{len(c['vid'])} clips -> {a.out}")


if __name__ == "__main__":
    main()
