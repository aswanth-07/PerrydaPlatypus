"""Build the submission bundle: <name>.zip holding src/ (predict(), its modules, requirements, README, models)
and results/ (predictions.jsonl, results.csv and the files they were computed with). Also writes
results/predictions.json, the same predictions as one JSON list, for scorers that read a single JSON file.

Usage: make_bundle.py [--name VIT_Vellore_PerrydaPlatypus] [--out .]
"""
import argparse
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="VIT_Vellore_PerrydaPlatypus")
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    res = ROOT / "results"
    rows = [json.loads(line) for line in open(res / "predictions.jsonl", encoding="utf-8") if line.strip()]
    (res / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")
    zpath = pathlib.Path(a.out) / f"{a.name}.zip"
    files = [p for d in ("src", "results") for p in sorted((ROOT / d).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, pathlib.Path(a.name) / p.relative_to(ROOT))
    print(f"{zpath}: {len(files)} files, {zpath.stat().st_size / 1e6:.1f} MB, {len(rows)} predictions")


if __name__ == "__main__":
    main()
