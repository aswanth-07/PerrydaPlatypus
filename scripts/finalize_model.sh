#!/usr/bin/env bash
# Ship one vocabulary: export the chosen checkpoints to INT8, measure robustness, copy into src/models/<vocab>.
# usage: [EXPORT_ARGS="--dynamic --ops Conv,Gemm,MatMul"] scripts/finalize_model.sh VOCAB SUBSET CKPT [CKPT...]
#        (VOCAB = full | core | union; EXPORT_ARGS picks the INT8 scheme, static Conv,Gemm by default)
# Robustness uses the 258 WLASL100 test clips degraded in pixels before MediaPipe (moderate and harsh sets),
# scored with this model's whole vocabulary, as for the models shipped before.
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-C:\\Users\\AASWAN~1\\AppData\\Local\\Temp\\xr2\\m\\Scripts\\python.exe}"
vocab=$1 subset=$2; shift 2
out="runs/deploy_new/$vocab"
extra=()
[ "$subset" -gt 2000 ] && extra=(--labels data/aslcitizen/union_labels.json)
mkdir -p "$out"
"$PY" -I scripts/export_v2.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "$out" "$@" --subset "$subset" "${extra[@]}" ${EXPORT_ARGS:-}
"$PY" -I scripts/eval_v2.py data/index/WLASL_v0.3.json data/kp_hol data/kp_rob100_mod "$@" --subset "$subset" --out "$out/robustness_moderate.json"
"$PY" -I scripts/eval_v2.py data/index/WLASL_v0.3.json data/kp_hol data/kp_rob100 "$@" --subset "$subset" --out "$out/robustness.json"
[ -f "runs/deploy/$vocab/labels.json" ] && [ "$subset" -gt 2000 ] && cp "runs/deploy/$vocab/labels.json" "$out/labels.json"  # readable union labels
mkdir -p "src/models/$vocab"
cp "$out/sign_int8.onnx" "$out/export.json" "$out/robustness.json" "$out/robustness_moderate.json" "src/models/$vocab/"
[ -f "$out/labels.json" ] && cp "$out/labels.json" "src/models/$vocab/"
[ -f "$out/sets.json" ] && cp "$out/sets.json" "src/models/$vocab/"   # same-sign set of each word (models exported with --set-rule)
echo "shipped $vocab from $*"
