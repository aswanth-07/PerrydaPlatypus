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
    def __init__(self, models, cfgs, face_flip):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.canon = [bool(c["canon"]) for c in cfgs]
        self.canon_rel = [bool(c.get("canon_rel", False)) for c in cfgs]
        self.register_buffer("perm", tv.flip_perm(face_flip), persistent=False)
        self.t_in = cfgs[0]["t_in"]

    def flip(self, x, w, s):
        xf = x.index_select(2, self.perm)
        xf = torch.cat([-xf[..., :1], xf[..., 1:]], -1)
        wf = torch.stack([w[:, :, 1], w[:, :, 0]], 2)
        wf = torch.cat([-wf[..., :1], wf[..., 1:]], -1)
        sf = torch.stack([s[..., 0], s[..., 2], s[..., 1], s[..., 3]], -1)
        return xf, wf, sf

    def forward(self, x, w, s):
        lengths = torch.full((x.shape[0],), self.t_in, dtype=torch.long, device=x.device)
        probs = 0
        for m, canon, rel in zip(self.models, self.canon, self.canon_rel):
            xi, wi, si = x, w, s
            if canon:
                left = tv.dominant_left(x, s, lengths, rel)[:, None, None, None]
                xf, wf, sf = self.flip(x, w, s)
                xi = torch.where(left, xf, x)
                wi = torch.where(left[..., None], wf, w)
                si = torch.where(left[:, :, :, 0], sf, s)
            probs = probs + torch.softmax(m(xi, wi, si), -1)
        return probs / len(self.models)


def batches(data, ids, t_in):
    out = []
    for i in ids.tolist():
        sel = torch.tensor([i])
        x, w, s = tv.sample_time(data["X"][sel], data["W"][sel], data["S"][sel], data["L"][sel], t_in, False)
        out.append(dict(points=x.numpy().astype(np.float32), world=w.numpy().astype(np.float32), seen=s.numpy().astype(np.float32)))
    return out


def run_onnx(path, feeds, y, threads=1):
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
    return tv.score(probs, y), dict(p50_ms=round(times[len(times) // 2], 3), p95_ms=round(times[int(0.95 * len(times))], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--calib", type=int, default=400)
    ap.add_argument("--ops", default="Conv,Gemm", help="op types quantized to INT8")
    ap.add_argument("--labels", default="", help="JSON label list for vocabularies beyond the WLASL index (union)")
    a = ap.parse_args()
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
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
    deploy = Deploy(models, [ck["cfg"] for ck in cks], cks[0].get("face_flip", torch.arange(62))).eval()
    t_in = deploy.t_in

    entries = json.load(open(a.index, encoding="utf-8"))
    cache = torch.load(a.cache, weights_only=False)
    rows = tv.select(cache, entries, a.subset)
    data = {k: cache[k] for k in ("X", "W", "S", "L")}
    test_ids, y = rows["test"]
    feeds = batches(data, test_ids, t_in)
    g = torch.Generator().manual_seed(0)
    calib_ids = rows["train"][0][torch.randperm(len(rows["train"][0]), generator=g)[:a.calib]]
    calib = batches(data, calib_ids, t_in)

    with torch.no_grad():
        torch_probs = torch.cat([deploy(*(torch.from_numpy(f[k]) for k in ("points", "world", "seen"))) for f in feeds])
    fp32, raw, int8 = out / "sign_fp32.onnx", out / "sign_pre.onnx", out / "sign_int8.onnx"
    ex = tuple(torch.from_numpy(feeds[0][k]) for k in ("points", "world", "seen"))
    torch.onnx.export(deploy, ex, str(fp32), input_names=["points", "world", "seen"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(raw))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            return next(self.it, None)

    # Only the weight-bearing ops go to INT8. The coordinate maths in front (palm frame, differences,
    # norms) and the activation-by-activation attention products lose 6 points top-1 when quantized.
    quantize_static(str(raw), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=a.ops.split(","))
    raw.unlink()
    res = dict(ckpts=a.ckpts, t_in=t_in, test_clips=len(y), torch=tv.score(torch_probs, y))
    for name, path in (("fp32", fp32), ("int8", int8)):
        sc, lat = run_onnx(path, feeds, y)
        res[name] = dict(size_mb=round(path.stat().st_size / 1e6, 3), test=sc, one_thread=lat)
    labels = json.loads(pathlib.Path(a.labels).read_text(encoding="utf-8")) if a.labels else [e["gloss"].upper() for e in entries[:a.subset]]
    labels = labels[:cks[0]["classes"]]
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (out / "export.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
