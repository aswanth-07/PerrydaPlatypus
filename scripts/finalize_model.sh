#!/usr/bin/env bash
# Ship the EchoGesture model: export one or more checkpoints (e.g. a 2-model diverse ensemble) to folded dynamic INT8
# and copy the artifacts into src/models, where predict() and the Android build read them.
# usage: scripts/finalize_model.sh CKPT [CKPT...]
#        EXPORT_ARGS defaults to "--dynamic --set-rule" (all 57 weighted MatMuls INT8, 24 depthwise+ECA convs FP32)
# Robustness is scored separately on all 2,878 test clips: scripts/eval_suite.py CKPT... --robust <moderate> <harsh>.
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
out="runs/deploy"
mkdir -p "$out"
ARGS="${EXPORT_ARGS:---dynamic --set-rule}"
"$PY" -I scripts/export.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "$out" "$@" --subset 2000 $ARGS
cp "$out/echogesture_int8.onnx" "$out/export.json" "$out/labels.json" src/models/
[ -f "$out/sets.json" ] && cp "$out/sets.json" src/models/   # same-sign set of each word (exports with --set-rule)
echo "shipped src/models/echogesture_int8.onnx from $*"
