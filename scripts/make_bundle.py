"""Build the standalone Track 2 submission ZIP: <name>.zip with src/, results/, conversion scripts/,
Dockerfile, requirements, README, AI disclosure and LICENSE directly at the archive top level.

Also writes results/predictions.json (an identical JSON array copy of results/predictions.jsonl) so
evaluators expecting either filename receive the exact same 2,878 records.

Usage: make_bundle.py [--name VIT_Vellore_PerrydaPlatypus] [--out .]
"""
import argparse
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Conversion script (scripts/export.py), its shell wrapper, and its full local Python import closure.
CONVERSION_SCRIPTS = (
    "scripts/export.py",
    "scripts/finalize_model.sh",
    "scripts/model.py",
    "scripts/train.py",
    "scripts/signsets.py",
    "scripts/contsign.py",
    "scripts/phonology.py",
)

ROOT_FILES = (
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "README.md",
    "AI_DISCLOSURE.md",
    "LICENSE",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="VIT_Vellore_PerrydaPlatypus")
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    res = ROOT / "results"
    rows = [json.loads(line) for line in open(res / "predictions.jsonl", encoding="utf-8") if line.strip()]
    clip_ids = [r["clip_id"] for r in rows]
    assert len(clip_ids) == len(set(clip_ids)), "duplicate clip_id in results/predictions.jsonl"
    (res / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")
    zpath = pathlib.Path(a.out) / f"{a.name}.zip"
    files = [p for d in ("src", "results") for p in sorted((ROOT / d).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    for rel in CONVERSION_SCRIPTS + ROOT_FILES:
        p = ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"required submission bundle file missing: {rel}")
        files.append(p)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{zpath}: {len(files)} files, {zpath.stat().st_size / 1e6:.2f} MB, {len(rows)} predictions")


if __name__ == "__main__":
    main()
