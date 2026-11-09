"""Write results.csv in the brief's format from measured files only.

Columns, as in the brief's team results sheet: metric, target, measured, unit, device, runtime, pass.
The six metrics the brief names come first, under its names (top1_accuracy, latency_p95, model_size,
accuracy_drop_robustness, avg_power, thermal_drift_10min); extra measured rows follow.

Robustness drops are measured with the PyTorch checkpoints the INT8 model was exported from (INT8 keeps 93-99%
top-1 agreement with them, see export.json). They are reported at two strengths of our degradations (moderate 0.4
and harsh 0.7), since the scorer's strength is not published; accuracy_drop_robustness uses the moderate set, and
the harsh drops are listed too.
Power and thermal drift need the phone; they are written as "not measured" rather than guessed.

Usage: results_csv.py <model_dir> <out.csv> [--unseen unseen.json] [--e2e predictions.summary.json]
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
    top1 = exp["int8"]["test"]["top1"]
    rows = [("top1_accuracy", ">=70", top1, "%", dev, rt, top1 >= 70)]
    rows.append(("latency_p95", "<=50", lat["p95_ms"], "ms", dev, rt + " (classifier, per clip)", lat["p95_ms"] <= 50))
    rows.append(("model_size", "<=20", round(size, 3), "MB", "", "INT8 ONNX classifier", size <= 20))
    if "moderate" in rob:
        worst = max(rob["moderate"][c]["drop"] for c in conds if c in rob["moderate"])
        worst_c = max((c for c in conds if c in rob["moderate"]), key=lambda c: rob["moderate"][c]["drop"])
        rows.append(("accuracy_drop_robustness", "<=5", round(worst, 2), "pt", "", f"{rob_rt}; worst moderate condition: {worst_c}", worst <= 5))
    rows += [("avg_power", "<=500", "not measured", "mW", "Android phone", "needs phone", "needs phone"),
             ("thermal_drift_10min", "<=20", "not measured", "%", "Android phone", "needs phone", "needs phone")]
    rows += [("top5_accuracy", "", exp["int8"]["test"]["top5"], "%", dev, rt, ""),
             ("test_clips", "", exp["test_clips"], "clips", "", f"WLASL official test split, {len(json.loads((d / 'labels.json').read_text(encoding='utf-8')))}-word vocabulary", ""),
             ("latency_p50", "", lat["p50_ms"], "ms", dev, rt + " (classifier, per clip)", ""),
             ("model_size_with_landmarker", "<=20", round(size + hol, 3), "MB", "", "INT8 classifier + MediaPipe Holistic task", size + hol <= 20)]
    for s in ("moderate", "harsh"):
        for c in conds:
            if s in rob and c in rob[s]:
                rows.append((f"robustness_drop_{s}_{c}", "<=5", rob[s][c]["drop"], "pt", "", rob_rt, rob[s][c]["drop"] <= 5))
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
