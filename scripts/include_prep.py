"""Unpack INCLUDE (Indian Sign Language, Zenodo record 4010759, CC BY 4.0) videos for the teach-a-sign test.

Each archive holds <Category>/<N. Word>/<MVI_xxxx>.MOV. Videos are written flat to <out>/videos/<cls>_<stem>.<ext>
(camera file numbers repeat across folders) and <out>/manifest.json lists one row per video in the shape the ASL
Citizen tools read (file, cls, gloss, signer), so scripts/extract_aslc.py, scripts/aslc_cache.py and
scripts/fewshot_eval.py work on it unchanged. Classes start at 2000, outside the 2,000-word vocabulary. The file
names carry no signer, so each row's signer is the clip itself: a few-shot split over this set is by clip, not by
signer, and its numbers are optimistic for that reason.

Usage: include_prep.py <zips_dir> <out_dir>
"""
import json
import pathlib
import re
import sys
import zipfile

VIDEO = (".mov", ".mp4", ".avi")


def main():
    zips, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    vids = out / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    members = []
    for z in sorted(zips.glob("*.zip")):
        if not zipfile.is_zipfile(z):
            print(f"skipped {z.name}: not a complete zip archive (download again to include it)")
            continue
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                p = pathlib.PurePosixPath(n)
                if p.suffix.lower() in VIDEO and len(p.parts) >= 3:
                    members.append((z, n, p.parts[0], p.parts[-2], p))
    words = sorted({(cat, re.sub(r"^\d+\.\s*", "", w).strip()) for _, _, cat, w, _ in members})
    cls_of = {cw: 2000 + k for k, cw in enumerate(words)}
    rows = []
    for z, n, cat, w, p in members:
        word = re.sub(r"^\d+\.\s*", "", w).strip()
        c = cls_of[(cat, word)]
        name = f"{c}_{p.stem}{p.suffix}"
        dst = vids / name
        if not dst.exists():
            with zipfile.ZipFile(z) as f, f.open(n) as src, open(dst.with_suffix(".part"), "wb") as o:
                while chunk := src.read(1 << 20):
                    o.write(chunk)
            dst.with_suffix(".part").replace(dst)
        rows.append(dict(file=name, cls=c, gloss=word.upper(), category=cat, signer=f"{c}_{p.stem}"))
    (out / "manifest.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    per = {}
    for r in rows:
        per[r["gloss"]] = per.get(r["gloss"], 0) + 1
    print(f"{len(rows)} videos, {len(words)} words -> {out}")
    print(json.dumps(per))


if __name__ == "__main__":
    main()
