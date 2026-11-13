"""Turn the class-alignment table into extra training clips.

ASL Citizen has 1,075 signs with no WLASL word of the same name. `analyses.py align` ran a 2,000-word model over
their clips and recorded, per ASL Citizen sign, the WLASL word it is called most often and how often (`agree`).
Where almost every clip of a sign is called the same word, the two are taken to be one sign under two names, and
the clips become training clips of that word. This writes a small cache holding only those clips and a manifest
giving each its WLASL class, for `train_v2.py --extra CACHE::MANIFEST`.

Usage: align_manifest.py TABLE.json ASLC_ALL_CACHE.pt MANIFEST.json WLASL_INDEX.json OUT_PREFIX [--agree 0.8] [--min-clips 10]
  MANIFEST   manifest_trainval.json for a measurement run (test signers stay unseen), manifest.json for deploy
  INDEX      data/index/WLASL_v0.3.json; class n is its n-th gloss
  writes     OUT_PREFIX.pt and OUT_PREFIX.json
"""
import argparse
import json
import pathlib

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("cache")
    ap.add_argument("manifest")
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--agree", type=float, default=0.8)
    ap.add_argument("--min-clips", type=int, default=10)
    a = ap.parse_args()

    classes = [e["gloss"].lower() for e in json.load(open(a.index, encoding="utf-8"))[:2000]]
    idx = {g: n for n, g in enumerate(classes)}
    table = json.load(open(a.table, encoding="utf-8"))
    to = {r["aslc"]: idx[r["wlasl"].lower()] for r in table
          if r["agree"] >= a.agree and r["clips"] >= a.min_clips and r["wlasl"].lower() in idx}

    manifest = json.load(open(a.manifest, encoding="utf-8"))
    keep = {"aslc_" + pathlib.Path(c["file"]).stem: c for c in manifest if c["gloss"] in to and c["cls"] >= 2000}
    cache = torch.load(a.cache, weights_only=False)
    rows = [n for n, v in enumerate(cache["vid"]) if v in keep]
    sel = torch.tensor(rows)
    small = {k: (v[sel] if torch.is_tensor(v) and len(v) == len(cache["vid"]) else v) for k, v in cache.items() if k != "vid"}
    small["vid"] = [cache["vid"][n] for n in rows]
    torch.save(small, a.out + ".pt")
    out = [dict(keep[v], cls=to[keep[v]["gloss"]], aligned_from=keep[v]["gloss"]) for v in small["vid"]]
    json.dump(out, open(a.out + ".json", "w", encoding="utf-8"))
    words = len({c["cls"] for c in out})
    print(f"aligned signs: {len(to)}  clips written: {len(out)}  WLASL words receiving clips: {words}")


if __name__ == "__main__":
    main()
