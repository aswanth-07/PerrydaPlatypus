"""Turn the union vocabulary's ASL Citizen gloss IDs (HADENOUGH, STETHOSCOPE2, HURDLE/TRIP1) into caption words
(HAD ENOUGH, STETHOSCOPE, HURDLE): word breaks from ASL-LEX entry IDs, variant numbers and alternatives dropped.
The first 2,000 labels (WLASL glosses) are kept as they are. Several classes may share a word; that is intended.

Usage: readable_labels.py <union_labels.json> <asllex signdata.csv> <out.json>
"""
import csv
import json
import re
import sys


def main():
    labels = json.load(open(sys.argv[1], encoding="utf-8"))
    rows = csv.DictReader(open(sys.argv[2], encoding="utf-8", errors="replace"))
    entry = {r["EntryID"].upper().replace("_", ""): r["EntryID"] for r in rows}

    def readable(g):
        g = entry.get(g, g).split("/")[0]
        return re.sub(r"\d+$", "", g).replace("_", " ").strip().upper() or g

    out = labels[:2000] + [readable(g) for g in labels[2000:]]
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"))
    print(f"{sum(a != b for a, b in zip(labels, out))} of {len(labels)} labels rewritten")


if __name__ == "__main__":
    main()
