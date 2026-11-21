#!/usr/bin/env bash
# Ship the live-caption model: the folded dynamic INT8 export of one or more checkpoints trained on windows of
# continuous signing (train.py --cont), with the decoder settings scripts/live_cont.py chose for it, into
# android/models/ (echogesture_live_int8.onnx and live.json). The app packs both; predict() does not use them.
# usage: scripts/export_live.sh CONT_JSON CKPT [CKPT...]
#        CONT_JSON = the --out file of live_cont.py; EXPORT_ARGS defaults to "--dynamic --exclude stem --set-rule"
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
cont=$1; shift
out="runs/deploy_live"
mkdir -p "$out"
ARGS="${EXPORT_ARGS:---dynamic --exclude stem --set-rule}"
"$PY" -I scripts/export.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "$out" "$@" --subset 2000 $ARGS
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
echo "shipped android/models/echogesture_live_int8.onnx from $*"
