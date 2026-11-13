#!/usr/bin/env bash
# Overnight chain of 2026-11-13, two runs at a time, safe to start again (a run with result.json is skipped):
#   1. two more measurement seeds of the plain recipe and of the clean-view teacher (--consist 1.0), so soups of
#      four plain runs, four teacher runs and mixes of both can be compared at equal size on unseen signers;
#   2. the eight shipping fine-tunes of the shipped soup (same seeds and starting points) with the teacher.
# Progress: runs/log_chain_cons.txt. Each run: runs/<name>/result.json, runs/log_<name>.txt.
set -u
cd "$(dirname "$0")/.."
M=runs/union2_measure_s73/best.pt
A=runs/union_deploy_s71/best.pt
B=runs/union2_deploy_s74/best.pt
queue=(
  "w2000_base_measure_s99 99 $M measure"
  "w2000_cons_measure_s99 99 $M measure --consist 1.0"
  "w2000_base_measure_s100 100 $M measure"
  "w2000_cons_measure_s100 100 $M measure --consist 1.0"
  "w2000_cons_d_s90 90 $A deploy --consist 1.0"
  "w2000_cons_d_s91 91 $A deploy --consist 1.0"
  "w2000_cons_d_s94 94 $A deploy --consist 1.0"
  "w2000_cons_d_s102 102 $A deploy --consist 1.0"
  "w2000_cons_d_s103 103 $A deploy --consist 1.0"
  "w2000_cons_d_s98 98 $B deploy --consist 1.0"
  "w2000_cons_d_s99 99 $B deploy --consist 1.0"
  "w2000_cons_d_s101 101 $B deploy --consist 1.0"
)
run() {
  name=$1 seed=$2 init=$3 kind=$4; shift 4
  if [ -f "runs/$name/result.json" ]; then echo "skip $name (done)"; return; fi
  echo "start $name $(date +%H:%M)"
  bash scripts/run_exp.sh "$name" "$seed" "$init" 2000 "$kind" 5e-4 30 "$@"
  echo "end $name $(date +%H:%M)"
}
n=0
for job in "${queue[@]}"; do
  # shellcheck disable=SC2086
  run $job >> runs/log_chain_cons.txt 2>&1 &
  n=$((n + 1))
  sleep 30
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
echo "chain done $(date)" >> runs/log_chain_cons.txt
