"""Print eval_suite.py results side by side, one column per run.

Usage: compare_suites.py NAME=runs/<run>/suite.json [NAME=... ...]
Rows: WLASL and unseen-signer scores, half-clip accuracy, and for every robustness set in the files each
condition's top-1 and drop, plus the mean of the four hard drops (low light, occlusion, low res, combined).
"""
import json
import sys

HARD = ("low_light", "occlusion", "low_res", "combined")


def main():
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    S = {k: json.load(open(v, encoding="utf-8")) for k, v in runs.items()}

    def row(label, f):
        cells = []
        for s in S.values():
            try:
                cells.append(f(s))
            except KeyError:
                cells.append("-")
        print(f"{label:34s}" + "".join(f"{c:>10}" for c in cells))

    print(f"{'':34s}" + "".join(f"{k:>10}" for k in S))
    for sec in ("wlasl", "unseen"):
        for m in ("top1", "top5", "rule_top1", "same_set_top1"):
            row(f"{sec} {m}", lambda s: s[sec][m])
    for sec in ("prefix_wlasl", "prefix_unseen"):
        for m in ("0.5", "0.7"):
            row(f"{sec} {m}", lambda s: s[sec][m])
    sets = sorted({k for s in S.values() for k in s if k.startswith("robust_")})
    for sec in sets:
        name = sec[len("robust_"):]
        row(f"{name} clips", lambda s: s[sec]["clips"])
        row(f"{name} clean top1", lambda s: s[sec]["clean"]["top1"])
        for c in HARD + ("angled", "moving_head"):
            row(f"{name} {c} top1", lambda s: s[sec][c]["top1"])
            row(f"{name} {c} drop", lambda s: s[sec][c]["drop"])
        row(f"{name} hard drops, mean", lambda s: round(sum(s[sec][c]["drop"] for c in HARD) / 4, 2))
        row(f"{name} hard top1, mean", lambda s: round(sum(s[sec][c]["top1"] for c in HARD) / 4, 2))
        row(f"{name} hard rule top1, mean", lambda s: round(sum(s[sec][c]["rule_top1"] for c in HARD) / 4, 2))


if __name__ == "__main__":
    main()
