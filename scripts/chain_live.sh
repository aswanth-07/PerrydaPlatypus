#!/usr/bin/env bash
# 13 Nov, Plan step 7: live-caption word error rate and delay with the app's spotter rules, tuned on val streams.
#   full_rule     shipped 2,000-word soup, answer rule as in the graph, threshold on the word's probability
#   full_setconf  the same, threshold on the top word's whole-set probability (Prediction.setConfidence)
#   core          the 100-word model the app's live mode starts with
set -u
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
O=runs/analysis_2026-11-13
mkdir -p $O
I=data/index/WLASL_v0.3.json
echo "full_rule $(date +%H:%M)"
"$PY" -I scripts/live_eval.py $I data/kp_hol runs/soup_mix16/best.pt --subset 2000 --set-rule --grid wide --out $O/live_full_rule.json
echo "full_setconf $(date +%H:%M)"
"$PY" -I scripts/live_eval.py $I data/kp_hol runs/soup_mix16/best.pt --subset 2000 --set-conf --grid wide --out $O/live_full_setconf.json
echo "core $(date +%H:%M)"
"$PY" -I scripts/live_eval.py $I data/kp_hol runs/core_occ_deploy_s111/best.pt runs/core_occ_deploy_s112/best.pt --subset 100 --grid wide --out $O/live_core.json
echo "live done $(date +%H:%M)"
