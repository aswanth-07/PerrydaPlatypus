"""Fetch web pages and save them as plain text, one file per name, for reading excerpts.

Usage: fetch_text.py <out_dir> name=url [name=url ...]
"""
import html.parser
import pathlib
import re
import sys
import urllib.request


class Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "svg", "annotation", "annotation-xml"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4", "section", "br", "table", "figure"):
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.parts.append(data)


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for item in sys.argv[2:]:
        name, url = item.split("=", 1)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            text = raw.decode("utf-8", "replace")
            if "<html" in text[:2000].lower() or "<!doctype" in text[:200].lower():
                parser = Text()
                parser.feed(text)
                text = "".join(parser.parts)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n", text)
            (out / f"{name}.txt").write_text(text, encoding="utf-8")
            print(f"{name}: {len(text)} chars")
        except Exception as e:
            print(f"{name}: FAILED {e}")


if __name__ == "__main__":
    main()
