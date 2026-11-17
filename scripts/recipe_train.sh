#!/usr/bin/env bash
# The fine-tunes behind the shipped EchoGesture model, two at a time; a run with result.json is skipped, so it is
# safe to start again. Each is scripts/run_exp.sh: 2,000 words, 30 epochs, learning rate 5e-4, from a model
# pretrained on 3,075 signs (WLASL + ASL Citizen):
#   measure  trained on ASL Citizen's train and validation signers only, then scored on its 11 unseen test
#            signers; plain recipe and clean-view teacher (--consist 1.0), compared at equal soup size;
#   deploy   trained on every clip; with the eight plain runs listed in recipe_soup.sh, these eight teacher runs
#            make the 16-member shipped soup.
# Progress: runs/log_recipe_train.txt. Each run: runs/<name>/result.json, runs/log_<name>.txt.
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
  run $job >> runs/log_recipe_train.txt 2>&1 &
  n=$((n + 1))
  sleep 30
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
echo "chain done $(date)" >> runs/log_recipe_train.txt
