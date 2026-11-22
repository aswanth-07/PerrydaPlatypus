#!/usr/bin/env bash
# Update android/models/live.json with the measured continuous-signing decoder settings for the unified
# EchoGesture model (src/models/echogesture_int8.onnx), optionally exporting checkpoints first.
# usage: scripts/export_live.sh CONT_JSON [CKPT...]
#        CONT_JSON = the --out file of live_cont.py; EXPORT_ARGS defaults to "--dynamic --set-rule"
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
cont=$1; shift
if [ "$#" -gt 0 ]; then
  scripts/finalize_model.sh "$@"
fi
"$PY" -I - src/models/export.json "$cont" android/models/live.json <<'PYEOF'
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
echo "updated android/models/live.json for unified src/models/echogesture_int8.onnx"
