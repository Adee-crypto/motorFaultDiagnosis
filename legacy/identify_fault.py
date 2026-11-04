"""
identify_fault.py  (v3 -- load-aware, trigger-based)
=====================================================
Rule-based identification -- no trained model needed.

    Stage A  Is it faulty?   accel-1 / microphone variance vs a healthy run
                             at the same speed AND load
    Stage B  Which fault?    trigger set v3 (6 faults; voltage unbalance and
                             all temperature triggers removed after the
                             loaded-data check), thresholds per load

One run:
    python3 identify_fault.py <sensor_csv> [condition_id] [--load 0|1]
Whole motor (recommended -- several speeds combined):
    python3 identify_fault.py <folder_with_that_motors_runs> --load 0|1

condition_id (1-8) and load (0 unloaded / 1 loaded) are read from file
names like K_A_5_1.csv if not given. Both are required: triggers apply
only at certain speeds, and thresholds and the healthy reference shift
with load.

Files used (working directory):
    trigger_calibration_load<load>.json   from calibrate_triggers.py
    healthy_feature_table_load<load>.csv  optional; otherwise the built-in
                                          healthy reference for that load
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

import fault_feature_extraction as F
from process_fault import load_data, estimate_sample_rate, TIME_CSV, CHANNEL_MAP, SPEED_LABELS


def parse_name(path):
    m = re.search(r"_(\d)_(\d)\.csv$", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def load_calibration(path):
    with open(path) as fh:
        cal = json.load(fh)
    for rec in cal.values():  # json turns int keys into strings
        rec["by_condition"] = {int(k): v for k, v in rec["by_condition"].items()}
    return cal


def healthy_row(condition_id, load):
    path = f"healthy_feature_table_load{load}.csv"
    if os.path.exists(path):
        h = pd.read_csv(path)
        if condition_id in h["condition_id"].values:
            return h[h.condition_id == condition_id].iloc[0].to_dict()
    return F.healthy_reference_for(condition_id, load)


def run_one(path, cid, load, cal, verbose=True):
    df = load_data(TIME_CSV, path)
    fs = estimate_sample_rate(df["Time"].values)
    vec = F.build_feature_vector(df, fs, channel_map=CHANNEL_MAP, condition_id=cid)
    faulty, evA = F.is_faulty(vec, cid, healthy_row(cid, load))
    scores, fired, best = F.score_triggers(vec, cid, cal)
    if verbose:
        print(f"\n=== {os.path.basename(path)}  condition {cid}: {SPEED_LABELS.get(cid, '?')}, "
              f"{'loaded' if load else 'unloaded'}")
        for k, v in evA.items():
            print(f"  Stage A {k:40s} {v:,.1f}x healthy")
        print(f"  -> {'FAULTY' if faulty else 'healthy'}")
        for tid, fault, feat, x, rule, hit in sorted(fired, key=lambda r: (r[1], r[0])):
            r = (f">= {rule['lo']:.4g}" if rule["op"] == ">=" else
                 f"<= {rule['hi']:.4g}" if rule["op"] == "<=" else f"in [{rule['lo']:.4g}, {rule['hi']:.4g}]")
            tier = F.TRIGGER_TIER.get(tid, "")
            print(f"  [{'x' if hit else ' '}] {tid:13s} {tier:9s} {fault:19s} {feat:32s} {x:>10.4g}  {r}")
    return faulty, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="one sensor CSV, or a folder with one motor's runs")
    ap.add_argument("condition_id", nargs="?", type=int)
    ap.add_argument("--load", type=int, choices=[0, 1])
    ap.add_argument("--cal", help="calibration json (default trigger_calibration_load<load>.json)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.path, "*.csv"))) if os.path.isdir(a.path) else [a.path]
    runs = []
    for f in files:
        cid, ld = parse_name(f)
        if not os.path.isdir(a.path) and a.condition_id:
            cid = a.condition_id
        ld = a.load if a.load is not None else ld
        if cid not in SPEED_LABELS or ld not in (0, 1):
            sys.exit(f"{f}: need condition_id (1-8) and --load 0|1 (or a name like X_X_3_1.csv)")
        runs.append((f, cid, ld))
    loads = {r[2] for r in runs}
    if len(loads) > 1:
        sys.exit("All runs of one motor must share the same load.")
    load = loads.pop()
    cal = load_calibration(a.cal or f"trigger_calibration_load{load}.json")

    total, n_faulty = {}, 0
    for f, cid, ld in runs:
        faulty, scores = run_one(f, cid, ld, cal)
        n_faulty += bool(faulty)
        for k, s in scores.items():
            total[k] = total.get(k, 0.0) + s["points"]

    print(f"\n##### Verdict over {len(runs)} run(s)")
    print(f"Stage A: {n_faulty}/{len(runs)} runs above the healthy level")
    if n_faulty == 0:
        print("Verdict: healthy")
        return
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v:5.2f} points")
    best = max(total, key=total.get) if total and max(total.values()) > 0 else None
    print(f"Verdict: {best or 'faulty - type unknown (no trigger fired)'}")
    if best and best not in ("faulty_bearing", "rotor_misalignment", "rotor_unbalance"):
        print("Note: this fault is identified only by 'candidate' triggers, which held in both loads in"
              " this dataset but are not yet confirmed. Treat as tentative.")
    if len(runs) < 8:
        print(f"Note: only {len(runs)} of 8 speeds given; single runs are right only about half the time.")


if __name__ == "__main__":
    main()
