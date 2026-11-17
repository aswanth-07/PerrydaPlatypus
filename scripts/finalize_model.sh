#!/usr/bin/env bash
# Ship the EchoGesture model: export the chosen checkpoint (a weight-averaged soup) to INT8 and copy it into
# src/models, where predict() and the Android build read it.
# usage: EXPORT_ARGS="--dynamic --ops Conv,Gemm,MatMul --exclude stem/Conv --set-rule" scripts/finalize_model.sh CKPT
# Robustness is scored separately on all 2,878 test clips: scripts/eval_suite.py CKPT --robust <moderate> <harsh>.
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
out="runs/deploy"
mkdir -p "$out"
"$PY" -I scripts/export.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "$out" "$@" --subset 2000 ${EXPORT_ARGS:-}
cp "$out/echogesture_int8.onnx" "$out/export.json" "$out/labels.json" src/models/
[ -f "$out/sets.json" ] && cp "$out/sets.json" src/models/   # same-sign set of each word (exports with --set-rule)
echo "shipped src/models/echogesture_int8.onnx from $*"
