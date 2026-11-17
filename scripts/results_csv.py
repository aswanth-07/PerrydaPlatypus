"""Write results.csv in the brief's format from measured files only.

Columns, as in the brief's team results sheet: metric, target, measured, unit, device, runtime, pass.
The six metrics the brief names come first, under its names (top1_accuracy, latency_p95, model_size,
accuracy_drop_robustness, avg_power, thermal_drift_10min); extra measured rows follow.

Robustness drops are measured with the PyTorch checkpoints the INT8 model was exported from (INT8 keeps 93-99%
top-1 agreement with them, see export.json). They are reported at two strengths of our degradations (moderate 0.4
and harsh 0.7), since the scorer's strength is not published; accuracy_drop_robustness uses the moderate set, and
the harsh drops are listed too.
Power and thermal drift need the phone; they are written as "not measured" rather than guessed.

With --suite (scripts/eval_suite.py output with --robust <moderate dir> <harsh dir>), the robustness rows come from
all 2,878 WLASL2000 test clips degraded at both strengths, top-1 with the same-sign answer rule as the shipped graph
answers.
With --phone (scripts/phone_bench_report.py output) and --phone-model (bench_model.jsonl from the app's benchmark),
the power, thermal and on-device latency rows come from the phone instead of saying "not measured".
With --vocab-subsets (scripts/vocab_subsets.py output) the same model's top-1 on WLASL's smaller standard word
lists is listed, and with --live-cont (scripts/live_cont.py --onnx output for the shipped live model) the app's
live-caption word error rate on the three kinds of signing stream; with --phone-voice (the app's live benchmark run with
--ez voice true) the delay from the camera frame that settles a word to the phone starting to speak it.
scripts/make_results.sh holds the whole command.
Usage: results_csv.py <model_dir> <out.csv> [--unseen unseen.json] [--e2e predictions.summary.json] [--suite suite.json]
       [--phone phone_summary.json] [--phone-model bench_model.jsonl] [--vocab-subsets f.json] [--live-cont f.json]
"""
import argparse
import csv
import json
import pathlib
import platform

DEVICE = "Laptop CPU (Intel Core Ultra 9 275HX), 1 thread"


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return f"Laptop CPU ({out}), 1 thread" if out else DEVICE
    except Exception:
        return DEVICE if platform.system() == "Windows" else f"{platform.processor()}, 1 thread"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deploy")
    ap.add_argument("out")
    ap.add_argument("--conditions", default="angled,moving_head,low_light,occlusion,low_res,combined")
    ap.add_argument("--unseen", default="", help="JSON with an unseen-signer score to list (optional)")
    ap.add_argument("--e2e", default="", help="predict_test.py summary JSON: end-to-end accuracy and timings (optional)")
    ap.add_argument("--runtime", default="ONNX Runtime 1.31.1, INT8, CPUExecutionProvider")
    ap.add_argument("--suite", default="", help="eval_suite.py JSON with both robustness strengths on 2,878 clips (optional)")
    ap.add_argument("--phone", default="", help="phone_bench_report.py summary of a 10-minute on-device live run (optional)")
    ap.add_argument("--phone-model", default="", help="the app's bench_model.jsonl: classifier latency on the phone (optional)")
    ap.add_argument("--phone-power", default="", help="phone_bench_report.py summary of a live run on battery: avg_power (optional)")
    ap.add_argument("--phone-clip", default="", help="the app's bench_clip jsonl: picked-video recognition time on the phone (optional)")
    ap.add_argument("--phone-voice", default="", help="the app's live benchmark jsonl run with --ez voice true: frame-to-speech delay (optional)")
    ap.add_argument("--phone-power-pairs", default="", help="phone_bench_report.py --pairs summary: the model's power, paired against a baseline (optional)")
    ap.add_argument("--phone-power-cond", default="on battery", help="the conditions of the paired power run, for the runtime column")
    ap.add_argument("--phone-model-power", default="",
                    help="baseline.json,classifier.json: screen-on baseline and classifier-at-live-rate power summaries (optional)")
    ap.add_argument("--phone-power-row", action="append", default=[],
                    help="metric=summary.json[=what]: an extra whole-phone power row from a run on battery (repeatable)")
    ap.add_argument("--vocab-subsets", default="", help="vocab_subsets.py JSON: top-1 on WLASL100/300/1000 with answers restricted (optional)")
    ap.add_argument("--alternatives", default="", help="alternatives_eval.py JSON: sign on screen with the caption's offered words (optional)")
    ap.add_argument("--live-cont", default="", help="live_cont.py --onnx JSON for the shipped live model: live-caption word error rate (optional)")
    a = ap.parse_args()
    d = pathlib.Path(a.deploy)
    exp = json.loads((d / "export.json").read_text())
    rob = {}
    size = (d / "echogesture_int8.onnx").stat().st_size / 1e6
    task = d / "holistic_landmarker.task"
    hol = task.stat().st_size / 1e6 if task.exists() else 0.0
    conds = a.conditions.split(",")
    lat = exp["int8"]["one_thread"]
    dev, rt = cpu_name(), a.runtime
    rob_rt = "PyTorch FP32, the checkpoints the INT8 model was exported from (accuracy, not timing)"
    if a.suite:
        su = json.loads(pathlib.Path(a.suite).read_text())
        rob, n_rob = {}, 0
        for k, v in su.items():
            if k.startswith("robust_"):
                strength = "moderate" if k.endswith("_mod") else "harsh"
                rob[strength] = {c: {"drop": r["rule_drop"]} for c, r in v.items() if isinstance(r, dict) and c != "clean"}
                n_rob = v["clips"]
        rob_rt += f", top-1 with the same-sign answer rule as shipped; {n_rob} WLASL2000 test clips per condition"
    top1 = exp["int8"]["test"]["top1"]
    rows = [("top1_accuracy", ">=70", top1, "%", dev, rt, top1 >= 70)]
    rows.append(("latency_p95", "<=50", lat["p95_ms"], "ms", dev, rt + " (classifier, per clip)", lat["p95_ms"] <= 50))
    rows.append(("model_size", "<=20", round(size, 3), "MB", "", "EchoGesture INT8 ONNX model (echogesture_int8.onnx)", size <= 20))
    if "moderate" in rob:
        worst = max(rob["moderate"][c]["drop"] for c in conds if c in rob["moderate"])
        worst_c = max((c for c in conds if c in rob["moderate"]), key=lambda c: rob["moderate"][c]["drop"])
        rows.append(("accuracy_drop_robustness", "<=5", round(worst, 2), "pt", "", f"{rob_rt}; worst moderate condition: {worst_c}", worst <= 5))
    ph = json.loads(pathlib.Path(a.phone).read_text()) if a.phone else None
    phone_dev = f"{ph['device']} ({ph['soc']}), Android {ph['android']}" if ph else "Android phone"
    def lanes_text(r):
        n = len(str(r.get("lanes", "gpu")).split("+"))
        return f"{n} lane{'s' if n > 1 else ''} on the {'GPU' if r['gpu_delegate'] else 'CPU'}"
    live_rt = (f"app live pipeline, signer video replayed like a {ph['fps_target']} fps camera: MediaPipe Holistic "
               f"({lanes_text(ph)}) + spotter + INT8 classifier, {int(ph['duration_s'])} s") if ph else ""
    pw = json.loads(pathlib.Path(a.phone_power).read_text()) if a.phone_power else ph
    if a.phone_power_pairs:
        pp = json.loads(pathlib.Path(a.phone_power_pairs).read_text())
        rows.append(("avg_power", "<=500", pp["classifier_phase_mw"], "mW", phone_dev,
                     f"the whole phone, screen on, {a.phone_power_cond}, while the EchoGesture INT8 model runs at the live caption rate "
                     f"(6 windows a second, 1 CPU thread); of this, the model's own share is {pp['classifier_increment_mw']:+d} mW "
                     f"(standard error {pp['classifier_increment_se_mw']} mW) over the {pp['baseline_mw']} mW screen-on baseline, "
                     f"{pp['rounds_used']} alternating baseline/model rounds; hand tracking for live video is in avg_power_live_pipeline",
                     pp["classifier_phase_mw"] <= 500))
        rows.append(("avg_power_model_increment", "", pp["classifier_increment_mw"], "mW", phone_dev,
                     f"the model's own power above the screen-on baseline, paired rounds ({a.phone_power_cond}); standard error "
                     f"{pp['classifier_increment_se_mw']} mW; CPU {pp['classifier_cpu_pct']}% of one core vs {pp['baseline_cpu_pct']}% at baseline", ""))
        rows.append(("power_baseline_screen_on", "", pp["baseline_mw"], "mW", phone_dev, f"the app open on a static screen, nothing tracked; {a.phone_power_cond}", ""))
        if pw and pw.get("avg_power_mw") is not None:
            rows.append(("avg_power_live_pipeline", "", pw["avg_power_mw"], "mW", phone_dev,
                         f"whole phone, screen on, {a.phone_power_cond}: the full live pipeline, MediaPipe Holistic on {lanes_text(pw)} at "
                         f"{pw['fps_mean']:.0f} fps + live captions, {pw['power_rows_on_battery'] * 10} s; "
                         f"{pw['avg_power_mw'] - pp['baseline_mw']:.0f} mW above the baseline, almost all of it hand tracking", ""))
    elif a.phone_model_power:
        base_p, model_p = (json.loads(pathlib.Path(x).read_text()) for x in a.phone_model_power.split(","))
        inc = model_p["avg_power_mw"] - base_p["avg_power_mw"]
        se = ((base_p["sd_mw"] ** 2 + model_p["sd_mw"] ** 2) / min(base_p["power_rows_on_battery"], model_p["power_rows_on_battery"])) ** 0.5
        rows.append(("avg_power", "<=500", round(inc), "mW", phone_dev,
                     f"INT8 classifier at the live caption rate (6 windows a second), on battery: whole phone "
                     f"{model_p['avg_power_mw']} mW minus the screen-on baseline {base_p['avg_power_mw']} mW, "
                     f"standard error about {se:.0f} mW; classifier {model_p.get('config', 'as shipped')}; the whole live pipeline is in avg_power_live_pipeline", inc <= 500))
        if pw and pw.get("avg_power_mw") is not None:
            rows.append(("avg_power_live_pipeline", "", pw["avg_power_mw"], "mW", phone_dev,
                         f"whole phone, screen on, on battery: {lanes_text(pw)} at {pw['fps_mean']:.0f} fps + spotter + "
                         f"classifier, {pw['power_rows_on_battery'] * 10} s; {pw['avg_power_mw'] - base_p['avg_power_mw']:.0f} mW above the screen-on baseline", ""))
    elif pw and pw.get("avg_power_mw") is not None:
        pw_rt = (f"app live pipeline on battery, {lanes_text(pw)}, {pw['power_rows_on_battery'] * 10} s; whole phone, "
                 "screen on, mean of |battery current| x voltage sampled every second")
        rows.append(("avg_power", "<=500", pw["avg_power_mw"], "mW", phone_dev, pw_rt, pw["avg_power_mw"] <= 500))
    else:
        rows.append(("avg_power", "<=500", "not measured", "mW", phone_dev, "needs a run on battery (USB charging hides the load)", "needs phone"))
    if ph:
        drift = ph["thermal_drift_10min_pct"]
        rows.append(("thermal_drift_10min", "<=20", drift, "%", phone_dev, live_rt + "; larger of throughput drop and Holistic p95 latency rise, first vs last minute", drift <= 20))
    else:
        rows.append(("thermal_drift_10min", "<=20", "not measured", "%", "Android phone", "needs phone", "needs phone"))
    rows += [("top5_accuracy", "", exp["int8"]["test"]["top5"], "%", dev, rt, ""),
             ("test_clips", "", exp["test_clips"], "clips", "", f"WLASL official test split, {len(json.loads((d / 'labels.json').read_text(encoding='utf-8')))}-word vocabulary", ""),
             ("latency_p50", "", lat["p50_ms"], "ms", dev, rt + " (classifier, per clip)", ""),
             ("model_size_with_landmarker", "<=20", round(size + hol, 3), "MB", "", "EchoGesture INT8 model + MediaPipe Holistic task", size + hol <= 20)]
    for s in ("moderate", "harsh"):
        for c in conds:
            if s in rob and c in rob[s]:
                rows.append((f"robustness_drop_{s}_{c}", "<=5", rob[s][c]["drop"], "pt", "", rob_rt, rob[s][c]["drop"] <= 5))
    if a.phone_model:
        pm = [json.loads(l) for l in open(a.phone_model, encoding="utf-8") if l.strip()]
        # The app's setting: 1 thread, workers not spinning (older logs have no "spin" field: 2 threads).
        two = next((r for r in pm if r["threads"] == 1 and r.get("spin") is False), None) or next((r for r in pm if r["threads"] == 2), pm[0])
        paced = f"; paced like the live spotter (6 a second, cores idle between calls) p95 {two['paced_p95_ms']:.1f} ms" if "paced_p95_ms" in two else ""
        rows.append(("latency_p95_phone", "<=50", round(two["p95_ms"], 2), "ms", phone_dev,
                     f"ONNX Runtime Android 1.31.1, INT8, CPU, {two['threads']} thread{'s' if two['threads'] > 1 else ''} "
                     f"(classifier, per clip, {two['runs']} runs back to back{paced})", two["p95_ms"] <= 50))
    if ph:
        rows += [("live_fps_phone", "", ph["fps_mean"], "fps", phone_dev, live_rt, ""),
                 ("holistic_p95_phone", "", ph["landmarks_p95_ms"], "ms", phone_dev, live_rt + "; frame to result", ""),
                 ("classifier_p95_phone_under_load", "<=50", ph["model_p95_ms"], "ms", phone_dev, live_rt, ph["model_p95_ms"] <= 50),
                 ("cpu_load_phone", "", ph["cpu_pct_of_one_core"], "% of one core", phone_dev, live_rt, ""),
                 ("memory_pss_phone", "", ph["pss_mb_max"], "MB", phone_dev, live_rt, "")]
        if ph.get("gpu_busy_pct_mean") is not None:
            rows.append(("gpu_busy_phone", "", ph["gpu_busy_pct_mean"], "%", phone_dev, live_rt, ""))
        if ph.get("ap_rise_c") is not None:
            rows.append(("soc_temp_rise_10min", "", ph["ap_rise_c"], "C", phone_dev, live_rt + f"; {ph['ap_c_start']} -> {ph['ap_c_end']} C", ""))
    for spec in a.phone_power_row:
        metric, path, *what = spec.split("=", 2)
        q = json.loads(pathlib.Path(path).read_text())
        if q.get("avg_power_mw") is not None:
            rows.append((metric, "", q["avg_power_mw"], "mW", phone_dev, (what[0] + "; " if what else "") +
                         f"{q['power_rows_on_battery'] * 10} s on battery, whole phone, screen on", ""))
    if a.phone_clip:
        pc = [json.loads(l) for l in open(a.phone_clip, encoding="utf-8") if l.strip()][-1]
        rows.append(("video_recognition_p50_phone", "", round(pc["total_p50_ms"], 0), "ms", phone_dev,
                     f"Videos tab on a picked clip: decode + 24 frames on 2 Holistic lanes ({'GPU' if pc['gpu'] else 'CPU'}) + classifier, {pc['runs']} warm runs", ""))
    if a.phone_voice:
        pv = [json.loads(l) for l in open(a.phone_voice, encoding="utf-8") if l.strip()]
        st = next(r for r in pv if r.get("event") == "start")
        v = next(r for r in pv if r.get("event") == "voice")
        rows.append(("voice_delay_p50_phone", "", v["voice_p50_ms"], "ms", phone_dev,
                     f"spoken captions: from the camera frame that settles a word to the phone's text-to-speech engine starting to speak it "
                     f"({'an installed voice, no network' if v['on_device'] else 'the default voice'}); {v['words_spoken']} words of a stream of chained "
                     f"test signs replayed like a {st['fps_target']} fps camera, 2,000 words; p90 {v['voice_p90_ms']} ms (a word waiting for "
                     f"the one before it to finish), worst {v['voice_max_ms']} ms; the word settles 200 ms after the hands come to rest; the "
                     "speaker's own output delay is not included", ""))
    if a.unseen:
        u = json.loads(pathlib.Path(a.unseen).read_text())
        rows.append(("top1_unseen_signers", ">=70", u["top1"], "%", "", "PyTorch FP32, matched runs that held these signers out; "
                     "ASL Citizen test split, signers in no training data", u["top1"] >= 70))
    if a.e2e:
        e = json.loads(pathlib.Path(a.e2e).read_text())
        shared = f"; {e['shards']} processes sharing the CPU" if e.get("shards", 1) > 1 else ""
        rows += [("top1_end_to_end", ">=70", e["top1"], "%", dev,
                  f"predict() on {e['clips']} raw test videos, {int(e['frames_p50'])} frames landmarked per clip (median)", e["top1"] >= 70),
                 ("top5_end_to_end", "", e["top5"], "%", dev, f"predict() on {e['clips']} raw test videos", ""),
                 ("pipeline_p95_per_clip", "", e["whole_clip_ms"]["p95"], "ms", dev, "decode + MediaPipe Holistic + classifier" + shared, ""),
                 ("keypoints_per_frame_p50", "", e["keypoints_per_frame_ms_p50"], "ms", dev, "MediaPipe Holistic, per landmarked frame" + shared, "")]
    if a.vocab_subsets:
        vs = json.loads(pathlib.Path(a.vocab_subsets).read_text())
        for size in (1000, 300, 100):
            r = vs[f"wlasl{size}"]["mask"]
            rows.append((f"top1_accuracy_wlasl{size}", ">=70", r["top1"], "%", dev,
                         f"{rt}; the same model file answering only from WLASL's {size:,} most common words (ECHO_VOCAB_FILE), "
                         f"{r['clips']} test clips of those words; top-5 {r['top5']}", r["top1"] >= 70))
    if a.alternatives:
        al = json.loads(pathlib.Path(a.alternatives).read_text())
        g = al["gated"]["0.8"]
        rows.append(("sign_on_screen_with_alternatives", "", g["sign_on_screen_3_shown"], "%", dev,
                     f"{rt}; the app's tap-to-correct: the caption's word plus two offered words (best word of each further same-sign set), "
                     f"offered when confidence in the shown sign is under 0.8 ({g['captions_with_chips_pct']}% of captions); the clip's sign is among "
                     f"the words on screen; top word alone {al['top1_sign']}%, always offered {al['sign_among_3_shown']}%; {al['clips']} whole test clips", ""))
    if a.live_cont:
        lc = json.loads(pathlib.Path(a.live_cont).read_text())
        what = (f"the app's live captions (the live INT8 model, one caption per sign) on {lc['continuous']['norest']['streams']} streams "
                f"of {lc['words']} chained held-out WLASL test clips per kind, {lc['subset']} words open; decoder settings cross-fitted on "
                "halves of the test clips")
        kinds = (("norest", "the hands never rest between signs"), ("mixed", "mixed"), ("rest", "a rest after most signs"))
        for k, name in kinds:
            r = lc["continuous"][k]
            rows.append((f"live_caption_word_error_rate_{k}", "", r["wer"], "%", "", f"{what}; {name}; words found {r['words_found']}%, "
                         f"false words per 100: {r['false_per_100_words']}; {r['wer_same_sign']}% when another name of the same sign counts", ""))
        r = lc["continuous"]["norest"]
        rows.append(("live_caption_delay_p50", "", r["delay_ms_median"], "ms", "", f"{what}; the hands never rest: when the caption settles "
                     f"relative to the sign's last frame (negative: before it ends); p90 {r['delay_ms_p90']} ms", ""))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for r in rows:
            w.writerow(r)
    print(pathlib.Path(a.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
