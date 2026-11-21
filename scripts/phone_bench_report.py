"""Summarise an on-device live benchmark (engine/Bench.kt) and its adb sampler (scripts/phone_sampler.sh) into the
brief's device metrics.

  throughput        frames tracked per second (Holistic + spotter + classifier on a replayed signer video)
  latency           Holistic frame-to-result p50/p95, classifier p50/p95 per window, both from the app's own clocks
  thermal drift     change from the first minute to the last minute of the run: throughput (drop, %) and Holistic
                    p95 latency (rise, %); the brief's thermal_drift_10min is the larger of the two
  temperatures      SoC (AP), skin and battery from the thermal service, start and end, and the rise
  load              mean process CPU (% of one core), mean GPU busy %, memory (PSS)
  power             only from rows on battery (plugged == 0): mean of |current| x voltage (1 s samples averaged per
                    row in newer logs); plugged-in rows are left out because the charger feeds the phone

Usage: phone_bench_report.py <live.jsonl> [<sampler.csv>] [--out summary.json]
       phone_bench_report.py --pairs <dir> [--out summary.json]

--pairs summarises the app's paired power run (Bench.powerPaired, `--es bench powerpair`): power_phase_pair<N>_baseline
and _classifier files in <dir>. Only rows on battery count, and only rounds whose two phases both have such rows; the
classifier's own power is the mean of the per-round differences (classifier phase minus its baseline), with its
standard error across rounds.
"""
import argparse
import csv
import json
import pathlib
import statistics as st


def pairs(d):
    def rows(f):
        return [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
    rounds = []
    for k in range(1, 100):
        fb, fc = d / f"power_phase_pair{k}_baseline.jsonl", d / f"power_phase_pair{k}_classifier.jsonl"
        if not (fb.exists() and fc.exists()):
            break
        b = [r for r in rows(fb) if "t_s" in r and r.get("plugged") == 0]
        c = [r for r in rows(fc) if "t_s" in r and r.get("plugged") == 0]
        if b and c:
            rounds.append(dict(round=k, baseline_mw=st.mean(r["power_mw"] for r in b), classifier_mw=st.mean(r["power_mw"] for r in c),
                               classifier_cpu_pct=st.mean(r["cpu_pct_of_one_core"] for r in c),
                               baseline_cpu_pct=st.mean(r["cpu_pct_of_one_core"] for r in b), calls_per_10s=st.mean(r["calls"] for r in c)))
    diffs = [r["classifier_mw"] - r["baseline_mw"] for r in rounds]
    return dict(rounds_used=len(rounds), rounds=[{k: round(v, 1) for k, v in r.items()} for r in rounds],
                baseline_mw=round(st.mean(r["baseline_mw"] for r in rounds)), classifier_phase_mw=round(st.mean(r["classifier_mw"] for r in rounds)),
                classifier_increment_mw=round(st.mean(diffs)), classifier_increment_se_mw=round(st.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else None,
                classifier_cpu_pct=round(st.mean(r["classifier_cpu_pct"] for r in rounds), 1), baseline_cpu_pct=round(st.mean(r["baseline_cpu_pct"] for r in rounds), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("live", nargs="?")
    ap.add_argument("--pairs", help="folder with the paired power run's power_phase_pair* files")
    ap.add_argument("sampler", nargs="?")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.pairs:
        out = pairs(pathlib.Path(a.pairs))
        print(json.dumps(out, indent=1))
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return
    rows = [json.loads(line) for line in open(a.live, encoding="utf-8") if line.strip()]
    start = next(r for r in rows if r.get("event") == "start")
    rep = [r for r in rows if "t_s" in r]
    if not rep:
        raise SystemExit("no periodic reports in " + a.live)
    first = [r for r in rep if r["t_s"] <= 65]
    last = [r for r in rep if r["t_s"] >= rep[-1]["t_s"] - 60]
    def m(rs, k):
        return st.mean(r[k] for r in rs)
    fps0, fps1 = m(first, "fps"), m(last, "fps")
    lat0, lat1 = m(first, "landmarks_p95_ms"), m(last, "landmarks_p95_ms")
    out = dict(
        device=start.get("device"), soc=start.get("soc"), android=start.get("android"), words=start.get("words", 2000),
        gpu_delegate=start.get("gpu", True), lanes=start.get("lanes", "gpu"), fps_target=start.get("fps_target"),
        frame=f'{start.get("frame_w")}x{start.get("frame_h")}', duration_s=rep[-1]["t_s"],
        fps_mean=round(m(rep, "fps"), 2), fps_first_min=round(fps0, 2), fps_last_min=round(fps1, 2),
        landmarks_p50_ms=round(st.median(r["landmarks_p50_ms"] for r in rep), 1),
        landmarks_p95_ms=round(st.median(r["landmarks_p95_ms"] for r in rep), 1),
        model_p50_ms=round(st.median([r["model_p50_ms"] for r in rep if r["windows"]] or [0.0]), 2),
        model_p95_ms=round(max([r["model_p95_ms"] for r in rep if r["windows"]] or [0.0]), 2),
        throughput_drift_pct=round(100 * (fps0 - fps1) / fps0, 1) if fps0 else None,
        latency_p95_drift_pct=round(100 * (lat1 - lat0) / lat0, 1) if lat0 else None,
        cpu_pct_of_one_core=round(m(rep, "cpu_pct_of_one_core"), 1),
        pss_mb_max=round(max(r["pss_mb"] for r in rep), 1),
        battery_c_start=rep[0]["battery_c"], battery_c_end=rep[-1]["battery_c"],
        thermal_status_max=max(r["thermal_status"] for r in rep),
        thermal_headroom_10s_end=round(rep[-1]["thermal_headroom_10s"], 3),
        plugged=any(r["plugged"] for r in rep),
    )
    out["thermal_drift_10min_pct"] = max(v for v in (out["throughput_drift_pct"], out["latency_p95_drift_pct"]) if v is not None)
    # Power only from rows on battery (plugged == 0): a charger feeds the phone, so the battery current is not the load.
    # Newer logs carry power_mw averaged from 1 s samples; older ones one current_now reading per row (uA on Samsung).
    batt = [r for r in rep if r["plugged"] == 0]
    def mw(r):
        if r.get("power_mw"):
            return r["power_mw"]
        i = abs(r["current_now"])
        return (i / 1e6 if i > 10_000 else i / 1e3) * r["voltage_mv"]
    out["avg_power_mw"] = round(st.mean(mw(r) for r in batt), 0) if batt else None
    out["power_rows_on_battery"] = len(batt)
    if a.sampler:
        s = [r for r in csv.DictReader(open(a.sampler, encoding="utf-8")) if r.get("ap_c")]
        def num(r, k):
            return float(r[k]) if r.get(k) not in (None, "") else None
        gpu = [num(r, "gpu_busy_pct") for r in s if num(r, "gpu_busy_pct") is not None]
        out.update(
            gpu_busy_pct_mean=round(st.mean(gpu[1:] or gpu), 1) if gpu else None,
            ap_c_start=num(s[0], "ap_c"), ap_c_end=num(s[-1], "ap_c"), ap_c_max=max(num(r, "ap_c") for r in s),
            skin_c_start=num(s[0], "skin_c"), skin_c_end=num(s[-1], "skin_c"), skin_c_max=max(num(r, "skin_c") for r in s),
        )
        out["ap_rise_c"] = round(out["ap_c_end"] - out["ap_c_start"], 1)
        out["skin_rise_c"] = round(out["skin_c_end"] - out["skin_c_start"], 1)
    print(json.dumps(out, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
