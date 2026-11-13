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
answers, instead of the 258-clip sets in the model folder.
With --phone (scripts/phone_bench_report.py output) and --phone-model (bench_model.jsonl from the app's benchmark),
the power, thermal and on-device latency rows come from the phone instead of saying "not measured".
Usage: results_csv.py <model_dir> <out.csv> [--unseen unseen.json] [--e2e predictions.summary.json] [--suite suite.json]
       [--phone phone_summary.json] [--phone-model bench_model.jsonl]
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
    a = ap.parse_args()
    d = pathlib.Path(a.deploy)
    exp = json.loads((d / "export.json").read_text())
    rob = {s: json.loads((d / f).read_text())["results"] for s, f in (("moderate", "robustness_moderate.json"), ("harsh", "robustness.json"))
           if (d / f).exists()}
    size = (d / "sign_int8.onnx").stat().st_size / 1e6
    task = d / "holistic_landmarker.task" if (d / "holistic_landmarker.task").exists() else d.parent / "holistic_landmarker.task"
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
    rows.append(("model_size", "<=20", round(size, 3), "MB", "", "INT8 ONNX classifier", size <= 20))
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
    if pw and pw.get("avg_power_mw") is not None:
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
             ("model_size_with_landmarker", "<=20", round(size + hol, 3), "MB", "", "INT8 classifier + MediaPipe Holistic task", size + hol <= 20)]
    for s in ("moderate", "harsh"):
        for c in conds:
            if s in rob and c in rob[s]:
                rows.append((f"robustness_drop_{s}_{c}", "<=5", rob[s][c]["drop"], "pt", "", rob_rt, rob[s][c]["drop"] <= 5))
    if a.phone_model:
        pm = [json.loads(l) for l in open(a.phone_model, encoding="utf-8") if l.strip()]
        two = next((r for r in pm if r["threads"] == 2), pm[0])
        rows.append(("latency_p95_phone", "<=50", round(two["p95_ms"], 2), "ms", phone_dev,
                     f"ONNX Runtime Android 1.31.1, INT8, CPU, {two['threads']} threads (classifier, per clip, {two['runs']} runs)", two["p95_ms"] <= 50))
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
    if a.phone_clip:
        pc = [json.loads(l) for l in open(a.phone_clip, encoding="utf-8") if l.strip()][-1]
        rows.append(("video_recognition_p50_phone", "", round(pc["total_p50_ms"], 0), "ms", phone_dev,
                     f"Videos tab on a picked clip: decode + 24 frames on 2 Holistic lanes ({'GPU' if pc['gpu'] else 'CPU'}) + classifier, {pc['runs']} warm runs", ""))
    if a.unseen:
        u = json.loads(pathlib.Path(a.unseen).read_text())
        rows.append(("top1_unseen_signers", ">=70", u["top1"], "%", "", "PyTorch FP32, matched runs that held these signers out; "
                     "ASL Citizen test split, signers in no training data", u["top1"] >= 70))
    if a.e2e:
        e = json.loads(pathlib.Path(a.e2e).read_text())
        shared = f"; {e['shards']} processes sharing the CPU" if e.get("shards", 1) > 1 else ""
        rows += [("top1_end_to_end", ">=70", e["top1"], "%", dev, f"predict() on {e['clips']} raw test videos", e["top1"] >= 70),
                 ("pipeline_p95_per_clip", "", e["whole_clip_ms"]["p95"], "ms", dev, "decode + MediaPipe Holistic + classifier" + shared, ""),
                 ("keypoints_per_frame_p50", "", e["keypoints_per_frame_ms_p50"], "ms", dev, "MediaPipe Holistic, per landmarked frame" + shared, "")]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for r in rows:
            w.writerow(r)
    print(pathlib.Path(a.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
