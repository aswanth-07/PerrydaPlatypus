#!/usr/bin/env bash
# One 100-word (core) training run in the shipped core recipe (as core_deploy2_*), warm-started from a 2000-word
# model, then (for measurement runs) the unseen-signer score at 100 words.
# usage: scripts/run_core.sh NAME SEED INIT measure|deploy [LR] [EPOCHS] [extra train_v2.py args...]
set -u
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-C:\\Users\\AASWAN~1\\AppData\\Local\\Temp\\xr2\\m\\Scripts\\python.exe}"
name=$1 seed=$2 init=$3 kind=$4 lr=${5:-5e-4} epochs=${6:-40}; shift $(( $# < 6 ? $# : 6 ))
if [ "$kind" = measure ]; then manifest=data/aslcitizen/manifest_trainval.json; else manifest=data/aslcitizen/manifest.json; fi
"$PY" -I scripts/train_v2.py data/index/WLASL_v0.3.json data/cache_hol2000w.pt "runs/$name" \
  --subset 100 --epochs "$epochs" --batch 64 --lr "$lr" --warmup 3 --canon --eval-every 10 \
  --phonology data/asllex/signdata.csv --trainval --aug-cache data/cache_aug100.pt --sparse 0.3 \
  --seed "$seed" --init "$init" --extra "data/cache_aslc100.pt::$manifest" "$@" > "runs/log_$name.txt" 2> "runs/err_$name.txt" \
  || { echo "TRAIN FAILED $name"; tail -5 "runs/err_$name.txt"; exit 1; }
if [ "$kind" = measure ]; then
  "$PY" -I scripts/eval_aslc.py data/cache_aslc100.pt data/aslcitizen/manifest_test.json "runs/$name/best.pt" --subset 100 \
    --out "runs/$name/unseen_100.json" >> "runs/log_$name.txt" 2>&1
fi
"$PY" -c "
import json, sys, pathlib
r = pathlib.Path('runs') / sys.argv[1]
t = json.loads((r / 'result.json').read_text())['test']
print(sys.argv[1], 'WLASL100 test:', {k: t.get(k) for k in ('n', 'top1', 'top5')})
for f in sorted(r.glob('unseen_*.json')):
    print(f.name, json.loads(f.read_text()))
" "$name"
