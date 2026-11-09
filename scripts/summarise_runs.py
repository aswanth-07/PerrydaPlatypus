"""Print one line per finished run, plus INT8 and robustness results where they exist.

Usage: summarise_runs.py <runs_dir>
"""
import json
import pathlib
import sys


def main():
    runs = pathlib.Path(sys.argv[1])
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        result = d / "result.json"
        if not result.exists():
            continue
        r = json.loads(result.read_text(encoding="utf-8"))
        print(f"{d.name}: classes={r['subset']} params={r['params']} train={r['train_clips']} single={r.get('single_frame')} "
              f"best_epoch={r['best_epoch']} secs={r['train_seconds']} val={r['val']['top1']:.1f} "
              f"test_n={r['test']['n']} top1={r['test']['top1']:.1f} top5={r['test']['top5']:.1f} class_avg={r['test']['class_avg']:.1f}")
        if r.get("test_vs_frames"):
            print("  vs frames:", {k: round(v, 1) for k, v in r["test_vs_frames"].items()})
        q = d / "int8.json"
        if q.exists():
            j = json.loads(q.read_text(encoding="utf-8"))
            print(f"  int8: {j['int8']['size_mb']} MB top1={j['int8']['test']['top1']:.1f} top5={j['int8']['test']['top5']:.1f} "
                  f"p50={j['int8']['one_thread']['p50_ms']} ms p95={j['int8']['one_thread']['p95_ms']} ms | fp32 top1={j['fp32']['test']['top1']:.1f}")
        rb = d / "robustness.json"
        if rb.exists():
            print("  robustness:", rb.read_text(encoding="utf-8").replace("\n", " "))


if __name__ == "__main__":
    main()
