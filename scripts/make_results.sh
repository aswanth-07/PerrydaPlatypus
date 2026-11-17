#!/usr/bin/env bash
# results/results.csv from the measured files, the whole command in one place. Run from anywhere; ECHO_PY is the
# training environment's Python 3.11. The inputs live in runs/ (kept out of Git):
#   runs/soup_mix16/suite_both.json        robustness on all 2,878 test clips at both strengths (scripts/eval_suite.py)
#   runs/msoup_mix8/unseen_2000.json       unseen signers, measurement runs that held them out
#   results/predictions.summary.json       predict() end to end on the 2,878 raw test videos (scripts/predict_test.py)
#   runs/phone_bench/final_1117/*          the phone, final build (scripts/phone_bench_report.py summaries):
#                                          10-minute thermal run, paired power run and live pipeline in airplane mode
#   runs/phone_bench/bench_model_final     the classifier's latency on the phone (the app's model bench)
#   runs/phone_bench/bench_clip_latest, voice_stream_25fps   picked-video time and sign-to-voice delay on the phone
#   runs/analysis_2026-11-15/*             WLASL sub-vocabularies (scripts/vocab_subsets.py), tap-to-correct
#                                          (scripts/alternatives_eval.py)
#   runs/analysis_2026-11-17/seg_full_int8 live captions of the INT8 live model (scripts/live_cont.py --onnx)
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"
B=runs/phone_bench
F=$B/final_1117
"$PY" -I scripts/results_csv.py src/models results/results.csv \
  --unseen runs/msoup_mix8/unseen_2000.json \
  --e2e results/predictions.summary.json \
  --suite runs/soup_mix16/suite_both.json \
  --phone $F/summary_thermal.json \
  --phone-model $B/bench_model_final.jsonl \
  --phone-power $F/summary_power_live_airplane.json \
  --phone-power-pairs $F/summary_power_pairs.json --phone-power-cond "on battery, airplane mode" \
  --phone-clip $B/bench_clip_latest.jsonl \
  --phone-voice $B/voice_stream_25fps.jsonl \
  --phone-power-row "power_idle_gated=$F/summary_power_live_30fps_cpu+cpu_blank_gated.json=nobody in view, signing-gated tracking rate (7.6 fps); mobile data on, 657 mW baseline in that session" \
  --vocab-subsets runs/analysis_2026-11-15/vocab_subsets.json \
  --alternatives runs/analysis_2026-11-15/alternatives.json \
  --live-cont runs/analysis_2026-11-17/seg_full_int8.json > /dev/null
echo "wrote results/results.csv"
