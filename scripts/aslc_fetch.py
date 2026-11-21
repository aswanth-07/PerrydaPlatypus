"""Fetch only the ASL Citizen clips whose gloss is in our WLASL vocabulary, straight out of the remote zip.

Each member is pulled with one HTTP range request (local header + compressed data) and inflated here, so
nothing else in the 46 GB archive is downloaded. Clips are fetched in vocabulary order (the most common
WLASL words first) and a manifest maps each saved clip to its WLASL class.

ASL Citizen (Microsoft Research, 2023) is licensed for non-commercial research use only.

Usage: aslc_fetch.py <zip_url> <splits_dir> <index.json> <signdata.csv> <out_dir> [--top 2000] [--threads 16]
"""
import argparse
import csv
import io
import json
import pathlib
import re
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import phonology as ph
from remote_zip import RangeFile


def mapping(splits_dir, entries, signdata, everything=False):
    """Clips mapped to WLASL classes; with everything=True, glosses outside WLASL get new classes after them
    (the union vocabulary), and the second return value lists the union's labels."""
    lex = list(csv.DictReader(open(signdata, encoding="latin-1")))
    code2entry = {r["Code"]: r["EntryID"] for r in lex}
    wl = {}
    for i, e in enumerate(entries):
        wl.setdefault(ph._key(e["gloss"]), i)
    labels = [e["gloss"].upper() for e in entries]
    new = {}
    out = []
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(pathlib.Path(splits_dir) / f"{split}.csv", encoding="utf-8")):
            g = r["Gloss"]
            cls = wl.get(ph._key(g), wl.get(ph._key(code2entry.get(r["ASL-LEX Code"], "")), wl.get(ph._key(re.sub(r"\d+$", "", g)))))
            if cls is None and everything:
                if g not in new:
                    new[g] = len(labels)
                    labels.append(g.upper())
                cls = new[g]
            if cls is not None:
                out.append(dict(file=r["Video file"], cls=cls, gloss=labels[cls], signer=r["Participant ID"], src_split=split))
    return sorted(out, key=lambda d: d["cls"]), labels


def fetch_member(url, info, dst, tries=5):
    head = 30 + len(info.filename.encode()) + 1024   # local header with a generous extra field allowance
    start, end = info.header_offset, info.header_offset + head + info.compress_size
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            sig, _, _, method, _, _, _, csize, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", raw[:30])
            assert sig == 0x04034b50
            body = raw[30 + nlen + xlen: 30 + nlen + xlen + info.compress_size]
            data = zlib.decompressobj(-15).decompress(body) if method == 8 else body
            assert len(data) == info.file_size
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(dst)
            return True
        except Exception as e:
            time.sleep(2 ** t)
            err = e
    print("failed", info.filename, err, flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("splits")
    ap.add_argument("index")
    ap.add_argument("signdata")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--all", action="store_true", help="every clip: words outside WLASL become new classes after it")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    entries = json.load(open(a.index, encoding="utf-8"))[:a.top]
    clips, labels = mapping(a.splits, entries, a.signdata, a.all)
    (out / "manifest.json").write_text(json.dumps(clips), encoding="utf-8")
    (out / "union_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(a.url), buffer_size=1 << 20))
    infos = {pathlib.PurePosixPath(i.filename).name: i for i in z.infolist()}
    todo = [c for c in clips if c["file"] in infos and not (out / "videos" / c["file"]).exists()]
    print(f"{len(clips)} clips in vocabulary, {len(todo)} to fetch", flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(a.threads) as pool:
        for _ok in pool.map(lambda c: fetch_member(a.url, infos[c["file"]], out / "videos" / c["file"]), todo):
            n += 1
            if n % 1000 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
