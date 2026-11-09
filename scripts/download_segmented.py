"""Resumable parallel download of one large file over HTTP range requests.

Chunks are written in place into a preallocated <out>.part; finished chunk numbers are recorded in
<out>.done so a rerun continues where it stopped. The file is renamed to <out> when every chunk is in.

Usage: download_segmented.py <url> <out> [--threads 16] [--chunk-mb 64]
"""
import argparse
import os
import pathlib
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(url, part, start, end, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "r+b") as f:
                f.seek(start)
                got = 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
            if got == end - start + 1:
                return got
        except Exception as e:  # network hiccup: back off and retry this chunk
            print(f"chunk {start}: {e} (try {t + 1})", flush=True)
        time.sleep(2 ** t)
    raise RuntimeError(f"chunk at {start} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    part, done_file = out.with_suffix(out.suffix + ".part"), out.with_suffix(out.suffix + ".done")
    total = size_of(a.url)
    chunk = a.chunk_mb << 20
    n = (total + chunk - 1) // chunk
    if not part.exists():
        with open(part, "wb") as f:
            f.truncate(total)
    done = set(int(x) for x in done_file.read_text().split()) if done_file.exists() else set()
    todo = [k for k in range(n) if k not in done]
    print(f"{total / 1e9:.2f} GB in {n} chunks, {len(todo)} to fetch, {a.threads} threads", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    fetched = [0]

    def job(k):
        start = k * chunk
        end = min(total, start + chunk) - 1
        got = fetch(a.url, part, start, end)
        with lock:
            done.add(k)
            with open(done_file, "a") as f:
                f.write(f"{k}\n")
            fetched[0] += got
            if len(done) % 20 == 0 or len(done) == n:
                rate = fetched[0] / max(time.time() - t0, 1) / 1e6
                print(f"{len(done)}/{n} chunks, {rate:.1f} MB/s, eta {(total - len(done) * chunk) / 1e6 / max(rate, 0.01) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(a.threads) as pool:
        list(pool.map(job, todo))
    if len(done) == n:
        os.replace(part, out)
        done_file.unlink()
        print("complete", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
