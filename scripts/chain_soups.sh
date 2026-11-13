#!/usr/bin/env bash
# 13 Nov: settle the clean-view teacher (IG-KZ-037) and score the shipping candidates on both robustness strengths.
#   1. soups of four measurement runs: plain, teacher, mixed 2+2, and all eight; full suite with the moderate set;
#   2. soup_mix16 = soupAll8's eight members + the eight teacher shipping runs; WLASL + both robustness strengths;
#   3. soupAll8 (candidate A) on both robustness strengths.
# Progress: runs/log_chain_soups.txt.
set -u
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"   # the training environment's Python 3.11
R=runs
P97=$R/w2000_fromunion2_measure_s97/best.pt; P98=$R/w2000_base_measure_s98/best.pt
P99=$R/w2000_base_measure_s99/best.pt; P100=$R/w2000_base_measure_s100/best.pt
T97=$R/w2000_cons_measure_s97/best.pt; T98=$R/w2000_cons_measure_s98/best.pt
T99=$R/w2000_cons_measure_s99/best.pt; T100=$R/w2000_cons_measure_s100/best.pt
"$PY" -I scripts/soup.py $R/msoup_plain4/best.pt $P97 $P98 $P99 $P100
"$PY" -I scripts/soup.py $R/msoup_teach4/best.pt $T97 $T98 $T99 $T100
"$PY" -I scripts/soup.py $R/msoup_mix22/best.pt $P97 $P98 $T99 $T100
"$PY" -I scripts/soup.py $R/msoup_mix8/best.pt $P97 $P98 $P99 $P100 $T97 $T98 $T99 $T100
for n in msoup_plain4 msoup_teach4 msoup_mix22 msoup_mix8; do
  echo "suite $n $(date +%H:%M)"
  "$PY" -I scripts/eval_suite.py $R/$n/best.pt --out $R/$n/suite_big.json --robust data/kp_rob2000_mod
done
"$PY" -I scripts/soup.py $R/soup_mix16/best.pt \
  $R/w2000_fromunion_s90/best.pt $R/w2000_fromunion_s91/best.pt $R/w2000_fromunion_s94/best.pt \
  $R/w2000_fromunion2_s98/best.pt $R/w2000_fromunion2_s99/best.pt $R/w2000_fromunion2_s101/best.pt \
  $R/w2000_fromunion_s102/best.pt $R/w2000_fromunion_s103/best.pt \
  $R/w2000_cons_d_s90/best.pt $R/w2000_cons_d_s91/best.pt $R/w2000_cons_d_s94/best.pt $R/w2000_cons_d_s102/best.pt \
  $R/w2000_cons_d_s103/best.pt $R/w2000_cons_d_s98/best.pt $R/w2000_cons_d_s99/best.pt $R/w2000_cons_d_s101/best.pt
for n in soup_mix16 soupAll8; do
  echo "suite $n both strengths $(date +%H:%M)"
  "$PY" -I scripts/eval_suite.py $R/$n/best.pt --out $R/$n/suite_both.json --skip unseen,prefix --robust data/kp_rob2000_mod data/kp_rob2000
done
echo "soups done $(date +%H:%M)"
