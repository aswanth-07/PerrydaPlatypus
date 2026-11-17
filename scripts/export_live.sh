#!/usr/bin/env bash
# Ship the live-caption model: the INT8 export of a checkpoint trained on windows of continuous signing
# (train.py --cont), with the decoder settings scripts/live_cont.py chose for it, into android/models/
# (echogesture_live_int8.onnx and live.json). The app packs both; predict() does not use them.
# usage: [EXPORT_ARGS="--dynamic --ops Conv,Gemm,MatMul --exclude stem/Conv --set-rule"] scripts/export_live.sh CKPT CONT_JSON
#        CONT_JSON = the --out file of live_cont.py for that checkpoint
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
ckpt=$1 cont=$2
out="runs/deploy_live"
mkdir -p "$out"
"$PY" -I scripts/export.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "$out" "$ckpt" --subset 2000 ${EXPORT_ARGS:-}
cp "$out/echogesture_int8.onnx" android/models/echogesture_live_int8.onnx
"$PY" -I - "$out/export.json" "$cont" android/models/live.json <<'PYEOF'
import json, sys
export, cont = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
s = cont["settings"]
frame_ms = 40   # the measurement streams are 25 fps: a stride in frames becomes a time for the phone
live = dict(window_ms=int(round(1000 * cont["window_s"])), stride_ms=int(s["stride"]) * frame_ms, lo=float(s["lo"]),
            gap=int(s["gap"]), emin=float(s["emin"]), commit=float(s["commit"]), tail_ms=int(s["tail_ms"]),
            ckpt=export["ckpts"], int8=export["int8"], none_class=export.get("none_class", False),
            measured=dict(streams_per_kind=cont["streams_per_kind"], words=cont["words"], continuous=cont.get("continuous")))
json.dump(live, open(sys.argv[3], "w"), indent=1)
print("live.json:", {k: live[k] for k in ("window_ms", "stride_ms", "lo", "gap", "emin", "commit", "tail_ms")})
PYEOF
echo "shipped android/models/echogesture_live_int8.onnx from $ckpt"
