"""Export a trained run to ONNX, quantize to INT8, and compare size, accuracy and latency.

Features go through the NumPy path here, so this also checks the path predict() will use.

Usage: export_int8.py <index.json> <kp_dir> <run_dir>
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as fx
import train_kp as tk


def clip_features(data, i):
    n = int(data["L"][i])
    pts = fx.resample(data["X"][i][:n].numpy(), tk.INPUT_T).astype(np.float32)
    seen = fx.resample(data["S"][i][:n].numpy(), tk.INPUT_T).astype(np.float32)
    return fx.make_features_np(pts, seen)[None]


class Reader(CalibrationDataReader):
    def __init__(self, data, ids):
        self.items = iter(ids)
        self.data = data

    def get_next(self):
        i = next(self.items, None)
        return None if i is None else {"features": clip_features(self.data, i)}


def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def score(sess, data, rows):
    ids, y = rows
    top1 = top5 = 0
    for i, c in zip(ids.tolist(), y.tolist()):
        logits = sess.run(None, {"features": clip_features(data, i)})[0][0]
        order = np.argsort(-logits)[:5]
        top1 += int(order[0] == c)
        top5 += int(c in order)
    return dict(n=len(y), top1=100 * top1 / len(y), top5=100 * top5 / len(y))


def latency(sess, sample, runs=400):
    times = []
    for k in range(runs + 20):
        t = time.perf_counter()
        sess.run(None, {"features": sample})
        if k >= 20:
            times.append(1000 * (time.perf_counter() - t))
    times.sort()
    return dict(p50_ms=round(times[len(times) // 2], 3), p95_ms=round(times[int(0.95 * len(times))], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("kp_dir")
    ap.add_argument("run")
    a = ap.parse_args()

    run = pathlib.Path(a.run)
    ck = torch.load(run / "best.pt", weights_only=False, map_location="cpu")
    model = tk.SignTCN(ck["classes"], ck["width"], ck["depth"])
    model.load_state_dict(ck["state"])
    model.eval()
    fp32 = run / "model_fp32.onnx"
    torch.onnx.export(model, (torch.zeros(1, tk.INPUT_T, fx.FEATURES),), str(fp32), opset_version=17,
                      input_names=["features"], output_names=["logits"], dynamo=False)

    entries = json.load(open(a.index, encoding="utf-8"))
    cache = tk.load_cache(entries, a.kp_dir)
    rows = tk.select(cache, entries, ck["classes"])
    rng = np.random.default_rng(7)
    calib = rng.choice(rows["train"][0].numpy(), size=min(400, len(rows["train"][0])), replace=False).tolist()

    prep = run / "model_prep.onnx"
    quant_pre_process(str(fp32), str(prep))
    int8 = run / "model_int8.onnx"
    quantize_static(str(prep), str(int8), Reader(cache, calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    prep.unlink()

    s_fp, s_q = session(fp32), session(int8)
    sample = clip_features(cache, rows["test"][0][0].item())
    result = dict(
        classes=ck["classes"], width=ck["width"], depth=ck["depth"],
        params=sum(p.numel() for p in model.parameters()),
        fp32=dict(size_mb=round(fp32.stat().st_size / 1e6, 3), test=score(s_fp, cache, rows["test"]), one_thread=latency(s_fp, sample)),
        int8=dict(size_mb=round(int8.stat().st_size / 1e6, 3), test=score(s_q, cache, rows["test"]), one_thread=latency(s_q, sample)),
    )
    (run / "int8.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
