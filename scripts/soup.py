"""Model soup: average the weights of SignNet checkpoints fine-tuned from the same warm start.

Usage: soup.py <out.pt> <ckpt.pt> <ckpt.pt> [...]
The result has one model's size and latency; it only works for checkpoints that share an initialisation.
"""
import pathlib
import sys

import torch


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    soup = dict(cks[0])
    state = {}
    for k, v in cks[0]["state"].items():
        if torch.is_floating_point(v):
            state[k] = sum(c["state"][k].float() for c in cks).div(len(cks)).to(v.dtype)
        else:
            state[k] = v
    soup["state"] = state
    soup["soup_of"] = [str(p) for p in paths]
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, out)
    print(f"soup of {len(paths)} -> {out}")


if __name__ == "__main__":
    main()
