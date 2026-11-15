#!/usr/bin/env bash
# results/results.csv from the measured files, the whole command in one place (it reproduced the table of
# 13 Nov byte for byte before the rows of 15 Nov were added). Run from anywhere; ECHO_PY is the training Python.
#   runs/phone_bench/*          summaries of the on-phone benchmarks (scripts/phone_bench_report.py); voice_stream_*
#                               is the app's own live benchmark file from a run with --ez voice true
#   runs/soup_mix16/suite_both  robustness on all 2,878 test clips at both strengths (scripts/eval_suite.py)
#   runs/msoup_mix8/unseen_2000 unseen signers, matched measurement runs
#   results/predictions.summary predict() end to end (scripts/predict_test.py)
#   runs/analysis_2026-11-15/*  sub-vocabularies (scripts/vocab_subsets.py), live captions (scripts/live_eval.py --compare)
set -eu
cd "$(dirname "$0")/.."
PY="${ECHO_PY:-python}"
B=runs/phone_bench
"$PY" -I scripts/results_csv.py src/models/full results/results.csv \
  --unseen runs/msoup_mix8/unseen_2000.json \
  --e2e results/predictions.summary.json \
  --suite runs/soup_mix16/suite_both.json \
  --phone $B/summary_thermal_cpu2.json \
  --phone-model $B/bench_model_spin2.jsonl \
  --phone-power "$B/summary_power_cpu+cpu.json" \
  --phone-clip $B/bench_clip_latest.jsonl \
  --phone-voice $B/voice_stream_25fps.jsonl \
  --phone-model-power $B/summary_power_phase_baseline.json,$B/summary_power_phase_classifier_6hz.json \
  --phone-power-row "power_idle_gated=$B/summary_power_cpu+cpu_blank_gated.json=nobody in view, signing-gated rate (7.6 fps)" \
  --phone-power-row "power_idle_ungated=$B/summary_power_cpu+cpu_blank.json=nobody in view, gating off (30 fps)" \
  --phone-power-row "power_live_one_gpu_tracker=$B/summary_power_gpu.json=the old setting: 1 Holistic tracker on the GPU (9.8 fps)" \
  --phone-power-row "power_baseline_screen_on=$B/summary_power_phase_baseline.json=the app open on a static screen, nothing tracked" \
  --vocab-subsets runs/analysis_2026-11-15/vocab_subsets.json \
  --alternatives runs/analysis_2026-11-15/alternatives.json \
  --live runs/analysis_2026-11-15/live_compare_60.json > /dev/null
cp results/results.csv src/models/full/results.csv
# results/predictions.json is the same predictions as one JSON list (what scripts/make_bundle.py also writes)
"$PY" -I -c "import json; rows = [json.loads(l) for l in open('results/predictions.jsonl', encoding='utf-8') if l.strip()]; open('results/predictions.json', 'w', encoding='utf-8').write(json.dumps(rows))"
echo "results/results.csv written"
