#!/usr/bin/env bash
# End to end: the shipped predict() (src/main.py, src/models/full) on all 2,878 raw WLASL2000 test videos in six
# shards side by side, merged into results/predictions.jsonl and its summary. About 40 minutes.
set -u
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
O=runs/e2e_full
mkdir -p $O
args="data/index/WLASL_v0.3.json data/wlasl21k/WLASL/videos data/wlasl_main/videos src/models/full"
for k in 0 1 2 3 4 5; do
  "$PY" -I scripts/predict_test.py $args $O/shard$k.jsonl --subset 2000 --shard $k/6 > $O/log_shard$k.txt 2>&1 &
done
wait
"$PY" -I scripts/predict_test.py $args results/predictions.jsonl --subset 2000 --merge $O/shard0.jsonl $O/shard1.jsonl $O/shard2.jsonl $O/shard3.jsonl $O/shard4.jsonl $O/shard5.jsonl > $O/log_merge.txt 2>&1
echo "e2e done $(date +%H:%M)" >> $O/log_merge.txt
