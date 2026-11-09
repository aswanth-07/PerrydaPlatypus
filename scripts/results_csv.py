"""Write results.csv in the brief's format (metric, value, unit, target, pass) from measured files only.

Robustness is reported at two strengths of our degradations (moderate 0.4 and harsh 0.7), since the scorer's
strength is not published; accuracy_drop_robustness uses the moderate set, and the harsh drops are listed too.
Power and thermal drift need the phone; they are written as "not measured" rather than guessed.

Usage: results_csv.py <deploy_dir> <out.csv> [--conditions angled,moving_head,low_light,occlusion,low_res,combined]
"""
import argparse
import csv
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deploy")
    ap.add_argument("out")
    ap.add_argument("--conditions", default="angled,moving_head,low_light,occlusion,low_res,combined")
    ap.add_argument("--unseen", default="", help="JSON with an unseen-signer score to list (optional)")
    a = ap.parse_args()
    d = pathlib.Path(a.deploy)
    exp = json.loads((d / "export.json").read_text())
    rob = {s: json.loads((d / f).read_text())["results"] for s, f in (("moderate", "robustness_moderate.json"), ("harsh", "robustness.json"))
           if (d / f).exists()}
    size = (d / "sign_int8.onnx").stat().st_size / 1e6
    task = d / "holistic_landmarker.task" if (d / "holistic_landmarker.task").exists() else d.parent / "holistic_landmarker.task"
    hol = task.stat().st_size / 1e6 if task.exists() else 0.0
    conds = [c for c in a.conditions.split(",")]
    lat = exp["int8"]["one_thread"]
    rows = [
        ("top1", exp["int8"]["test"]["top1"], "%", ">=70", exp["int8"]["test"]["top1"] >= 70),
        ("top5", exp["int8"]["test"]["top5"], "%", "", ""),
        ("test_clips", exp["test_clips"], "clips", "", "WLASL official test split"),
        ("latency_p95", lat["p95_ms"], "ms", "<=50", lat["p95_ms"] <= 50),
        ("model_size", round(size, 3), "MB", "<=25", size <= 25),
        ("model_size_with_landmarker", round(size + hol, 3), "MB", "<=25", size + hol <= 25),
    ]
    if "moderate" in rob:
        worst = max(rob["moderate"][c]["drop"] for c in conds if c in rob["moderate"])
        rows.append(("accuracy_drop_robustness", round(worst, 2), "pt", "<=5", worst <= 5))
    rows += [("avg_power", "not measured", "mW", "<=500", "needs phone"),
             ("thermal_drift_10min", "not measured", "%", "<=20", "needs phone")]
    for s in ("moderate", "harsh"):
        for c in conds:
            if s in rob and c in rob[s]:
                rows.append((f"robustness_drop_{s}_{c}", rob[s][c]["drop"], "pt", "<=5", rob[s][c]["drop"] <= 5))
    if a.unseen:
        u = json.loads(pathlib.Path(a.unseen).read_text())
        rows.append(("top1_unseen_signers", u["top1"], "%", ">=70", u["top1"] >= 70))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit", "target", "pass"])
        for r in rows:
            w.writerow(r)
    print(pathlib.Path(a.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
