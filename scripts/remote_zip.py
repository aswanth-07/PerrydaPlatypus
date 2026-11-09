"""Read members of a remote zip over HTTP range requests (no full download): list it, or pull small files.

Usage: remote_zip.py <url> list [pattern]
       remote_zip.py <url> get <member> <out_file>
"""
import io
import re
import sys
import urllib.request
import zipfile


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    url, cmd = sys.argv[1], sys.argv[2]
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20))
    if cmd == "list":
        pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
        names = z.infolist()
        print(f"{len(names)} members")
        for i in names:
            if pat is None or pat.search(i.filename):
                print(i.filename, i.file_size, i.compress_type)
    elif cmd == "get":
        with open(sys.argv[4], "wb") as f:
            f.write(z.read(sys.argv[3]))
        print("saved", sys.argv[4])


if __name__ == "__main__":
    main()
