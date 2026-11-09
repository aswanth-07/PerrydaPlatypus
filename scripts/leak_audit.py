"""WLASL test clips that share an overlapping source segment with a train or val clip, and top-1 with and
without them.

Some WLASL test entries are frame ranges of the same source video (same URL, overlapping frames; -1 = to the end)
as train/val entries, so a model may have seen those very frames, usually under another gloss (WLASL's synonyms).
This counts them and scores a predictions.jsonl (predict_test.py output) on all test clips, on the overlapping
ones and on the rest.

Usage: leak_audit.py <index.json> <predictions.jsonl> [--subset 2000] [--out leak.json]
"""
import argparse
import collections
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("predictions")
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    entries = json.load(open(a.index, encoding="utf-8"))[:a.subset]
    seen = collections.defaultdict(list)   # url -> (start, end, gloss) of train/val clips cut from it
    for e in entries:
        for i in e["instances"]:
            if i["split"] in ("train", "val"):
                seen[i["url"]].append((i["frame_start"], i["frame_end"], e["gloss"].upper()))

    def end(f):
        return float("inf") if f < 0 else f

    truth, kind = {}, {}
    for e in entries:
        for i in e["instances"]:
            if i["split"] == "test":
                g = e["gloss"].upper()
                truth[i["video_id"]] = g
                hit = {o[2] for o in seen.get(i["url"], []) if max(i["frame_start"], o[0]) <= min(end(i["frame_end"]), end(o[1]))}
                kind[i["video_id"]] = ("same_gloss" if g in hit else "other_gloss") if hit else "clean"
    hits = collections.Counter()
    count = collections.Counter()
    for line in open(a.predictions, encoding="utf-8"):
        r = json.loads(line)
        k = kind.get(r["clip_id"])
        if k is None:
            continue
        ok = r["prediction"] == truth[r["clip_id"]]
        for group in ("all", k, "overlap" if k != "clean" else None):
            if group:
                count[group] += 1
                hits[group] += ok
    res = {g: dict(clips=count[g], top1=round(100 * hits[g] / count[g], 2)) for g in count}
    print(json.dumps(res, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
