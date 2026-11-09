"""File size and one-thread CPU speed of candidate caption models.

Weights are random: this measures size and compute only, and says nothing about accuracy.

Usage: caption_bench.py <out_dir> [--pin]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as fx
import train_kp as tk


class CTCNet(nn.Module):
    """Conv encoder with one output per frame, as a streaming gloss recogniser would use."""

    def __init__(self, vocab, width=192, depth=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(fx.FEATURES, width, 1, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[tk.TBlock(width, 2 ** (i % 3), 0.0) for i in range(depth)])
        self.head = nn.Conv1d(width, vocab, 1)

    def forward(self, features):
        return self.head(self.blocks(self.stem(features.transpose(1, 2))))


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.out = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.dh).transpose(1, 2)

    def forward(self, query, source, mask=None):
        q, k, v = self.split(self.q(query)), self.split(self.k(source)), self.split(self.v(source))
        score = torch.matmul(q, k.transpose(2, 3)) / (self.dh ** 0.5)
        if mask is not None:
            score = score + mask
        y = torch.matmul(F.softmax(score, dim=-1), v).transpose(1, 2)
        return self.out(y.reshape(y.shape[0], y.shape[1], self.heads * self.dh))


class Layer(nn.Module):
    def __init__(self, d, heads, ff, cross):
        super().__init__()
        self.a1, self.n1 = Attention(d, heads), nn.LayerNorm(d)
        self.a2, self.n2 = (Attention(d, heads), nn.LayerNorm(d)) if cross else (None, None)
        self.f1, self.f2, self.n3 = nn.Linear(d, ff), nn.Linear(ff, d), nn.LayerNorm(d)

    def forward(self, x, mem=None, mask=None):
        n = self.n1(x)
        x = x + self.a1(n, n, mask)
        if self.a2 is not None:
            x = x + self.a2(self.n2(x), mem)
        return x + self.f2(F.relu(self.f1(self.n3(x))))


class Encoder(nn.Module):
    def __init__(self, d, layers, frames, heads=4, ff=None):
        super().__init__()
        self.proj = nn.Linear(fx.FEATURES, d)
        self.pos = nn.Parameter(torch.zeros(1, frames, d))
        self.layers = nn.ModuleList(Layer(d, heads, ff or 2 * d, False) for _ in range(layers))

    def forward(self, features):
        x = self.proj(features) + self.pos[:, :features.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Transformer decoder run without a cache: each new token re-reads every earlier token."""

    def __init__(self, vocab, d, layers, heads=4, ff=None, max_len=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        self.layers = nn.ModuleList(Layer(d, heads, ff or 2 * d, True) for _ in range(layers))
        self.out = nn.Linear(d, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.full((max_len, max_len), -1e4), 1)[None, None])

    def forward(self, tokens, memory):
        n = tokens.shape[1]
        x = self.emb(tokens) + self.pos[:, :n]
        m = self.mask[:, :, :n, :n]
        for layer in self.layers:
            x = layer(x, memory, m)
        return self.out(x[:, -1])


class GRUDecoder(nn.Module):
    """One decoding step of an attention GRU, written with Linear layers so that every weight quantizes."""

    def __init__(self, vocab, hidden, emb=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.attn = nn.Linear(hidden, hidden)
        n = emb + hidden
        self.xz, self.hz = nn.Linear(n, hidden), nn.Linear(hidden, hidden)
        self.xr, self.hr = nn.Linear(n, hidden), nn.Linear(hidden, hidden)
        self.xh, self.hh = nn.Linear(n, hidden), nn.Linear(hidden, hidden)
        self.bottleneck = nn.Linear(2 * hidden, emb)
        self.out = nn.Linear(emb, vocab, bias=False)

    def forward(self, token, h, memory):
        w = F.softmax(torch.matmul(memory, self.attn(h).unsqueeze(-1)).squeeze(-1), dim=-1)
        ctx = torch.matmul(w.unsqueeze(1), memory).squeeze(1)
        x = torch.cat([self.emb(token).squeeze(1), ctx], dim=-1)
        z = torch.sigmoid(self.xz(x) + self.hz(h))
        r = torch.sigmoid(self.xr(x) + self.hr(h))
        h2 = (1 - z) * h + z * torch.tanh(self.xh(x) + self.hh(r * h))
        return self.out(F.relu(self.bottleneck(torch.cat([h2, ctx], dim=-1)))), h2


class Reader(CalibrationDataReader):
    def __init__(self, sample):
        self.rows = iter(sample)

    def get_next(self):
        return next(self.rows, None)


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timed(fn, runs=60, warm=5):
    times = []
    for k in range(runs + warm):
        t = time.perf_counter()
        fn()
        if k >= warm:
            times.append(1000 * (time.perf_counter() - t))
    times.sort()
    return round(times[len(times) // 2], 2), round(times[int(0.95 * len(times))], 2)


def export(model, args, names, outputs, axes, path):
    model.eval()
    torch.onnx.export(model, args, str(path), opset_version=17, input_names=names, output_names=outputs,
                      dynamic_axes=axes, dynamo=False)


def conv_case(name, model, frames, out):
    fp = out / f"{name}.onnx"
    q = out / f"{name}_int8.onnx"
    export(model, (torch.zeros(1, frames, fx.FEATURES),), ["features"], ["logits"], {}, fp)
    rng = np.random.default_rng(0)
    sample = [{"features": rng.normal(0, 1, (1, frames, fx.FEATURES)).astype(np.float32)} for _ in range(12)]
    quantize_static(str(fp), str(q), Reader(sample), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    s = session(q)
    p50, p95 = timed(lambda: s.run(None, sample[0]))
    return dict(design=name, quant="static QDQ", params=sum(p.numel() for p in model.parameters()),
                fp32_mb=round(fp.stat().st_size / 1e6, 2), int8_mb=round(q.stat().st_size / 1e6, 2),
                unit=f"one {frames}-frame window", p50_ms=p50, p95_ms=p95)


def dynamic(fp, q):
    quantize_dynamic(str(fp), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "Gather"])


def encoder_case(name, model, frames, out):
    fp = out / f"{name}.onnx"
    q = out / f"{name}_int8.onnx"
    export(model, (torch.zeros(1, frames, fx.FEATURES),), ["features"], ["memory"], {}, fp)
    dynamic(fp, q)
    s = session(q)
    x = np.random.default_rng(0).normal(0, 1, (1, frames, fx.FEATURES)).astype(np.float32)
    p50, p95 = timed(lambda: s.run(None, {"features": x}), runs=30)
    return dict(design=name, quant="dynamic", params=sum(p.numel() for p in model.parameters()),
                fp32_mb=round(fp.stat().st_size / 1e6, 2), int8_mb=round(q.stat().st_size / 1e6, 2),
                unit=f"one {frames}-frame window", p50_ms=p50, p95_ms=p95)


def decoder_case(name, model, d, frames, out, sentence=15):
    fp = out / f"{name}.onnx"
    q = out / f"{name}_int8.onnx"
    export(model, (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, frames, d)), ["tokens", "memory"], ["logits"],
           {"tokens": {1: "n"}}, fp)
    dynamic(fp, q)
    s = session(q)
    mem = np.random.default_rng(0).normal(0, 1, (1, frames, d)).astype(np.float32)

    def whole():
        for n in range(1, sentence + 1):
            s.run(None, {"tokens": np.ones((1, n), np.int64), "memory": mem})

    p50, p95 = timed(whole, runs=8, warm=1)
    return dict(design=name, quant="dynamic", params=sum(p.numel() for p in model.parameters()),
                fp32_mb=round(fp.stat().st_size / 1e6, 2), int8_mb=round(q.stat().st_size / 1e6, 2),
                unit=f"{sentence}-token sentence, no cache", p50_ms=p50, p95_ms=p95)


def gru_case(name, model, hidden, frames, out, sentence=15):
    fp = out / f"{name}.onnx"
    q = out / f"{name}_int8.onnx"
    export(model, (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, hidden), torch.zeros(1, frames, hidden)),
           ["token", "h", "memory"], ["logits", "h2"], {}, fp)
    dynamic(fp, q)
    s = session(q)
    mem = np.random.default_rng(0).normal(0, 1, (1, frames, hidden)).astype(np.float32)

    def whole():
        h = np.zeros((1, hidden), np.float32)
        tok = np.ones((1, 1), np.int64)
        for _ in range(sentence):
            logits, h = s.run(None, {"token": tok, "h": h, "memory": mem})
            tok = np.array([[int(logits[0].argmax())]], np.int64)

    p50, p95 = timed(whole, runs=30)
    return dict(design=name, quant="dynamic", params=sum(p.numel() for p in model.parameters()),
                fp32_mb=round(fp.stat().st_size / 1e6, 2), int8_mb=round(q.stat().st_size / 1e6, 2),
                unit=f"{sentence}-token sentence", p50_ms=p50, p95_ms=p95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.pin:
        psutil.Process().cpu_affinity([0])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [
        lambda: conv_case("classifier_2001_w192", tk.SignTCN(2001, 192, 6), tk.INPUT_T, out),
        lambda: conv_case("classifier_2001_w256", tk.SignTCN(2001, 256, 6), tk.INPUT_T, out),
        lambda: conv_case("ctc_2000_w192", CTCNet(2000, 192), 96, out),
        lambda: conv_case("ctc_5000_w192", CTCNet(5000, 192), 96, out),
        lambda: conv_case("ctc_8000_w256", CTCNet(8000, 256), 96, out),
        lambda: encoder_case("tf_encoder_d192_l4", Encoder(192, 4, 96), 96, out),
        lambda: encoder_case("tf_encoder_d256_l4", Encoder(256, 4, 96), 96, out),
        lambda: gru_case("gru_decoder_h256_v4000", GRUDecoder(4000, 256), 256, 12, out),
        lambda: gru_case("gru_decoder_h256_v8000", GRUDecoder(8000, 256), 256, 12, out),
        lambda: gru_case("gru_decoder_h384_v8000", GRUDecoder(8000, 384), 384, 12, out),
        lambda: decoder_case("tf_decoder_d192_l2_v8000", Decoder(8000, 192, 2), 192, 96, out),
        lambda: decoder_case("tf_decoder_d256_l2_v8000", Decoder(8000, 256, 2), 256, 96, out),
        lambda: decoder_case("tf_decoder_d256_l4_v8000", Decoder(8000, 256, 4), 256, 96, out),
        lambda: encoder_case("t5small_like_encoder", Encoder(512, 6, 128, heads=8, ff=2048), 128, out),
        lambda: decoder_case("t5small_like_decoder_v32128", Decoder(32128, 512, 6, heads=8, ff=2048), 512, 128, out),
    ]
    rows = []
    for case in cases:
        try:
            row = case()
        except Exception as e:
            row = dict(design=getattr(case, "__name__", "?"), error=repr(e)[:300])
        rows.append(row)
        print(json.dumps(row), flush=True)
    for p in out.glob("*.onnx"):
        p.unlink()
    (out / "caption_bench.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
