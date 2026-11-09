"""Per-gloss phonological labels from ASL-LEX 2.0 (CC BY 4.0, Sehyr et al. 2021) for auxiliary training heads.

WLASL glosses are matched to ASL-LEX EntryIDs by lower-cased English gloss (EntryID with any _N suffix and
underscores removed), as in Kezar et al., EACL 2023. Unmatched glosses get -1 and are ignored by the loss.

Usage (report coverage): phonology.py <signdata.csv> <index.json> [--top 2000]
"""
import argparse
import csv
import json
import re

TYPES = ["Handshape", "MinorLocation", "SignType", "Movement", "SelectedFingers", "Flexion",
         "MajorLocation", "NonDominantHandshape", "RepeatedMovement"]


def _key(s):
    s = s.lower().strip()
    s = re.sub(r"_\d+$", "", s)
    return re.sub(r"[_\s]+", " ", s)


def load(signdata, glosses, types=TYPES):
    """-> (labels: list per type of per-class value ids, -1 where unknown; sizes: values per type; matched count)"""
    rows = list(csv.DictReader(open(signdata, encoding="latin-1")))
    by_key = {}
    for r in rows:
        by_key.setdefault(_key(r["EntryID"]), r)       # the first (most common) sense wins
        if r.get("LemmaID"):
            by_key.setdefault(_key(r["LemmaID"]), r)
    vocab = {t: sorted({r[f"{t}.2.0"] for r in rows if r.get(f"{t}.2.0") not in (None, "", "NA")}) for t in types}
    labels = {t: [] for t in types}
    matched = 0
    for g in glosses:
        r = by_key.get(_key(g))
        matched += r is not None
        for t in types:
            v = r.get(f"{t}.2.0") if r else None
            labels[t].append(vocab[t].index(v) if v in vocab[t] else -1)
    return labels, {t: len(vocab[t]) for t in types}, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signdata")
    ap.add_argument("index")
    ap.add_argument("--top", type=int, default=2000)
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))[:a.top]
    labels, sizes, matched = load(a.signdata, [e["gloss"] for e in entries])
    clips = [len(e["instances"]) for e in entries]
    covered = sum(c for c, h in zip(clips, labels["Handshape"]) if h >= 0)
    print(f"{matched}/{len(entries)} glosses matched; {covered}/{sum(clips)} clips covered")
    print({t: (s, sum(v >= 0 for v in labels[t])) for t, s in sizes.items()})


if __name__ == "__main__":
    main()
