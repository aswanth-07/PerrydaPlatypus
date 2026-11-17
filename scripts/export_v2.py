"""Export SignNet checkpoints to ONNX (FP32 and static INT8 QDQ) for ONNX Runtime on Android, check that
INT8 keeps test accuracy, and time it on one CPU thread.

The exported graph takes the clip already resampled to t_in steps and does everything else itself:
dominant-hand flip (if the model was trained with it), features, network, and averaging over an ensemble.
Inputs:  points [1, T, 119, 3], world [1, T, 2, 21, 3], seen [1, T, 4]   Output: probs [1, classes]

Usage: export_v2.py <index.json> <cache.pt> <out_dir> <ckpt.pt> [<ckpt.pt> ...] [--subset 100] [--calib 400]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_v2 as mv
import train_v2 as tv


class Deploy(nn.Module):
    """Outputs: probs (what predict() answers from: the same-sign answer rule applied when the model has one),
    set_probs (per gloss, the probability of its whole same-sign set: the confidence the captions use),
    embedding (the pooled features, for signs the user teaches on the device), raw_probs (the network's own
    probabilities, before the rule). A live model (trained with a "no sign" class, train_v2.py --cont) also returns
    none [1, 1], the probability that the window is centred between two signs; its word outputs then sum to
    1 - none."""

    def __init__(self, models, cfgs, face_flip, rule=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.canon = [bool(c["canon"]) for c in cfgs]
        self.canon_rel = [bool(c.get("canon_rel", False)) for c in cfgs]
        self.register_buffer("perm", tv.flip_perm(face_flip), persistent=False)
        self.t_in = cfgs[0]["t_in"]
        self.rule = rule
        self.none = bool(cfgs[0].get("none_class"))

    def flip(self, x, w, s):
        xf = x.index_select(2, self.perm)
        xf = torch.cat([-xf[..., :1], xf[..., 1:]], -1)
        wf = torch.stack([w[:, :, 1], w[:, :, 0]], 2)
        wf = torch.cat([-wf[..., :1], wf[..., 1:]], -1)
        sf = torch.stack([s[..., 0], s[..., 2], s[..., 1], s[..., 3]], -1)
        return xf, wf, sf

    def forward(self, x, w, s):
        lengths = torch.full((x.shape[0],), self.t_in, dtype=torch.long, device=x.device)
        probs, emb = 0, 0
        for m, canon, rel in zip(self.models, self.canon, self.canon_rel):
            xi, wi, si = x, w, s
            if canon:
                left = tv.dominant_left(x, s, lengths, rel)[:, None, None, None]
                xf, wf, sf = self.flip(x, w, s)
                xi = torch.where(left, xf, x)
                wi = torch.where(left[..., None], wf, w)
                si = torch.where(left[:, :, :, 0], sf, s)
            e = m.embed(xi, wi, si)
            probs = probs + torch.softmax(m.head(e), -1)
            emb = emb + e
        raw, emb = probs / len(self.models), emb / len(self.models)
        none = ()
        if self.none:   # the last class is "no sign": it leaves the word outputs and comes out on its own
            raw, none = raw[:, :-1], (raw[:, -1:],)
        if self.rule is None:
            return (raw, raw, emb, raw) + none
        ruled, mass = self.rule(raw)
        return (ruled, mass, emb, raw) + none


def batches(data, ids, t_in):
    out = []
    for i in ids.tolist():
        sel = torch.tensor([i])
        x, w, s = tv.sample_time(data["X"][sel], data["W"][sel], data["S"][sel], data["L"][sel], t_in, False)
        out.append(dict(points=x.numpy().astype(np.float32), world=w.numpy().astype(np.float32), seen=s.numpy().astype(np.float32)))
    return out


def run_onnx(path, feeds, y, threads=1, reference=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    probs, times = [], []
    for f in feeds:
        t = time.perf_counter()
        probs.append(sess.run(None, f)[0])
        times.append(1000 * (time.perf_counter() - t))
    probs = torch.from_numpy(np.concatenate(probs))
    times = sorted(times[5:])
    sc = tv.score(probs, y)
    if reference is not None:  # label-free fidelity: how often the ONNX model's top-1 equals the PyTorch model's
        sc["agree_top1_with_torch"] = round(100 * float((probs.argmax(1) == reference.argmax(1)).float().mean()), 2)
    return sc, dict(p50_ms=round(times[len(times) // 2], 3), p95_ms=round(times[int(0.95 * len(times))], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--calib", type=int, default=400)
    ap.add_argument("--ops", default="Conv,Gemm", help="op types quantized to INT8")
    ap.add_argument("--calib-method", default="MinMax", choices=["MinMax", "Entropy", "Percentile"],
                    help="how activation ranges are set from the calibration clips")
    ap.add_argument("--act-type", default="QInt8", choices=["QInt8", "QUInt8"], help="activation integer type (static)")
    ap.add_argument("--dynamic", action="store_true",
                    help="weight-only INT8 (activations quantized on the fly) instead of static calibration")
    ap.add_argument("--exclude", default="", help="comma-separated substrings of node names kept in FP32 (e.g. stem/Conv,eca)")
    ap.add_argument("--labels", default="", help="JSON label list for vocabularies beyond the WLASL index (union)")
    ap.add_argument("--set-rule", action="store_true",
                    help="put the same-sign answer rule (signsets.py, sets from train+val listings) inside the graph")
    a = ap.parse_args()
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cks = [torch.load(c, weights_only=False) for c in a.ckpts]
    models = []
    for ck in cks:
        m = mv.build(ck["cfg"], ck["classes"])
        m.load_state_dict(ck["state"])
        if isinstance(m.head, mv.CosineHead):
            m.head = m.head.folded()   # a plain Linear on normalised input, so it is quantized as a Gemm
        models.append(m.eval())
    entries = json.load(open(a.index, encoding="utf-8"))
    rule, sets_info = None, None
    if a.set_rule:
        import signsets
        n_idx, n_cls = min(a.subset, len(entries)), cks[0]["classes"] - (1 if cks[0]["cfg"].get("none_class") else 0)
        set_of, links = signsets.same_sign_sets(entries, n_idx)
        counts = signsets.wlasl_counts(entries, n_idx)
        # Classes outside the WLASL index (a union vocabulary) are sets of their own with a neutral weight.
        set_of = torch.cat([set_of, int(set_of.max()) + 1 + torch.arange(n_cls - n_idx)])
        counts = torch.cat([counts, torch.ones(n_cls - n_idx)])
        rule = signsets.SetRule(set_of, counts)
        sets_info = dict(links=links, sets_of_two_or_more=int((torch.bincount(set_of) > 1).sum()),
                         gamma=signsets.GAMMA, beta=signsets.BETA, set_of=set_of.tolist())
    deploy = Deploy(models, [ck["cfg"] for ck in cks], cks[0].get("face_flip", torch.arange(62)), rule).eval()
    t_in = deploy.t_in

    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, a.subset)
    data = {k: cache[k] for k in ("X", "W", "S", "L")}
    test_ids, y = rows["test"]
    feeds = batches(data, test_ids, t_in)
    g = torch.Generator().manual_seed(0)
    calib_ids = rows["train"][0][torch.randperm(len(rows["train"][0]), generator=g)[:a.calib]]
    calib = batches(data, calib_ids, t_in)

    with torch.no_grad():
        outs = [deploy(*(torch.from_numpy(f[k]) for k in ("points", "world", "seen"))) for f in feeds]
        torch_probs = torch.cat([o[0] for o in outs])
        torch_raw = torch.cat([o[3] for o in outs])
    fp32, raw, int8 = out / "sign_fp32.onnx", out / "sign_pre.onnx", out / "sign_int8.onnx"
    ex = tuple(torch.from_numpy(feeds[0][k]) for k in ("points", "world", "seen"))
    torch.onnx.export(deploy, ex, str(fp32), input_names=["points", "world", "seen"],
                      output_names=["probs", "set_probs", "embedding", "raw_probs"] + (["none"] if deploy.none else []),
                      opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(raw))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            return next(self.it, None)

    # Only the weight-bearing ops go to INT8. The coordinate maths in front (palm frame, differences,
    # norms) and the activation-by-activation attention products lose 6 points top-1 when quantized.
    import onnx
    keep = [k for k in a.exclude.split(",") if k]
    exclude = [n.name for n in onnx.load(str(raw)).graph.node if any(k in n.name for k in keep)]
    if a.dynamic:
        quantize_dynamic(str(raw), str(int8), per_channel=True, weight_type=QuantType.QInt8,
                         op_types_to_quantize=a.ops.split(","), nodes_to_exclude=exclude)
    else:
        quantize_static(str(raw), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=getattr(QuantType, a.act_type), weight_type=QuantType.QInt8,
                        op_types_to_quantize=a.ops.split(","), calibrate_method=getattr(CalibrationMethod, a.calib_method),
                        nodes_to_exclude=exclude)
    raw.unlink()
    res = dict(ckpts=a.ckpts, t_in=t_in, test_clips=len(y), none_class=deploy.none,
               int8_scheme=dict(dynamic=a.dynamic, ops=a.ops, fp32_nodes=exclude),
               torch=tv.score(torch_probs, y))
    if sets_info is not None:  # the network's own answer, before the rule, for the record
        res["torch_before_rule"] = tv.score(torch_raw, y)
        res["same_sign"] = {k: v for k, v in sets_info.items() if k != "set_of"}
        (out / "sets.json").write_text(json.dumps(sets_info["set_of"]), encoding="utf-8")
    for name, path in (("fp32", fp32), ("int8", int8)):
        sc, lat = run_onnx(path, feeds, y, reference=torch_probs)
        res[name] = dict(size_mb=round(path.stat().st_size / 1e6, 3), test=sc, one_thread=lat)
    labels = json.loads(pathlib.Path(a.labels).read_text(encoding="utf-8")) if a.labels else [e["gloss"].upper() for e in entries[:a.subset]]
    labels = labels[:cks[0]["classes"] - (1 if deploy.none else 0)]
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (out / "export.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
