#!/usr/bin/env bash
# One training run in the recipe the shipped models use, then (for measurement runs) the unseen-signer score.
# usage: scripts/run_exp.sh NAME SEED INIT SUBSET measure|deploy LR EPOCHS [extra train.py args...]
#   measure: ASL Citizen train/val signers only, so its 11 test signers stay unseen; scored on them afterwards
#   deploy:  every ASL Citizen clip (test signers too), for shipping
set -u
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-C:\\Users\\AASWAN~1\\AppData\\Local\\Temp\\xr2\\m\\Scripts\\python.exe}"
name=$1 seed=$2 init=$3 subset=$4 kind=$5 lr=$6 epochs=$7; shift 7
if [ "$subset" -gt 2000 ]; then cache=data/cache_aslc_all.pt; else cache=data/cache_aslc2000.pt; fi
if [ "$kind" = measure ]; then manifest=data/aslcitizen/manifest_trainval.json; else manifest=data/aslcitizen/manifest.json; fi
"$PY" -I scripts/train.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "runs/$name" \
  --subset "$subset" --epochs "$epochs" --batch 128 --lr "$lr" --warmup 2 --canon --eval-every 10 \
  --phonology data/asllex/signdata.csv --trainval --aug-cache data/cache_aug2000.pt --sparse 0.3 --t-max 48 \
  --seed "$seed" --init "$init" --extra "$cache::$manifest" "$@" > "runs/log_$name.txt" 2> "runs/err_$name.txt" || { echo "TRAIN FAILED $name"; tail -5 "runs/err_$name.txt"; exit 1; }
if [ "$kind" = measure ]; then
  for n in 2000 $([ "$subset" -gt 2000 ] && echo "$subset"); do
    "$PY" -I scripts/eval_aslc.py "$cache" data/aslcitizen/manifest_test.json "runs/$name/best.pt" --subset "$n" \
      --out "runs/$name/unseen_$n.json" >> "runs/log_$name.txt" 2>&1
  done
fi
"$PY" - "$name" <<'EOF'
import json, sys, pathlib
r = pathlib.Path("runs") / sys.argv[1]
d = json.loads((r / "result.json").read_text())
t = d.get("test", d)
print(sys.argv[1], "WLASL test:", {k: t.get(k) for k in ("n", "top1", "top5")} if isinstance(t, dict) else t)
for f in sorted(r.glob("unseen_*.json")):
    print(f.name, json.loads(f.read_text()))
EOF
