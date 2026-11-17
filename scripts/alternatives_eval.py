"""What the app's "or:" chips are worth: how often the sign that was made is on screen when the caption shows the
top word and, beside it, the best word of each further same-sign set (names of the top word's own set are already
shown as "same sign").

The shipped INT8 model is scored on the WLASL test split of its vocabulary. A shown word counts as right when it
is the clip's label or a name of the same sign (sets.json; built from train and validation listings only).
--gates lists confidences under which the chips are shown at all: a caption the model is sure of stays clean, and
the table says what each gate costs.

Usage: alternatives_eval.py [--model-dir src/models] [--cache data/cache_hol2000w.pt]
       [--index data/index/WLASL_v0.3.json] [--out runs/analysis/alternatives.json]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # normalise, extract, holistic_kp
import export as ex  # noqa: E402
import train as tv  # noqa: E402


def shown(order, set_of, k):
    """The top word, then the best word of each further same-sign set, k words in all (the app's rule)."""
    out, sets = [], set()
    for j in order:
        if set_of[j] in sets:
            continue
        sets.add(set_of[j])
        out.append(int(j))
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "src/models"))
    ap.add_argument("--cache", default=str(ROOT / "data/cache_hol2000w.pt"))
    ap.add_argument("--index", default=str(ROOT / "data/index/WLASL_v0.3.json"))
    ap.add_argument("--gates", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    md = pathlib.Path(a.model_dir)
    labels = json.loads((md / "labels.json").read_text(encoding="utf-8"))
    set_of = np.array(json.loads((md / "sets.json").read_text()))
    entries = json.load(open(a.index, encoding="utf-8"))
    cache = torch.load(a.cache, weights_only=False)
    ids, y = tv.select(cache, entries, len(labels))["test"]
    y = y.numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(md / "echogesture_int8.onnx"), so, providers=["CPUExecutionProvider"])
    outs = [sess.run(["probs", "set_probs"], f) for f in ex.batches({k: cache[k] for k in ("X", "W", "S", "L")}, ids, 64)]
    probs = np.concatenate([o[0] for o in outs])
    set_probs = np.concatenate([o[1] for o in outs])
    order = np.argsort(-probs, 1)
    n = len(y)
    sure = np.array([set_probs[i, order[i, 0]] for i in range(n)])     # the model's confidence in the top SIGN
    hit = {k: np.array([set_of[y[i]] in {set_of[j] for j in shown(order[i], set_of, k)} for i in range(n)]) for k in (1, 2, 3)}
    res = {
        "clips": n,
        "top1_word": round(100 * float(np.mean(order[:, 0] == y)), 2),
        "top1_sign": round(100 * float(hit[1].mean()), 2),
        "sign_among_2_shown": round(100 * float(hit[2].mean()), 2),
        "sign_among_3_shown": round(100 * float(hit[3].mean()), 2),
        "gated": {},
    }
    for g in [float(x) for x in a.gates.split(",") if x]:
        chips = sure < g                                              # alternatives are offered only below the gate
        res["gated"][str(g)] = {
            "captions_with_chips_pct": round(100 * float(chips.mean()), 1),
            "sign_on_screen_2_shown": round(100 * float(np.where(chips, hit[2], hit[1]).mean()), 2),
            "sign_on_screen_3_shown": round(100 * float(np.where(chips, hit[3], hit[1]).mean()), 2),
            "top_right_when_no_chips": round(100 * float(hit[1][~chips].mean()), 2) if (~chips).any() else None,
        }
    print(json.dumps(res, indent=1))
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
