"""
calibrate_final.py
===================
Fits the thresholds of the final trigger set for ONE load condition and
writes them to a JSON file that predict_motor.py reads.

    python3 calibrate_final.py <tables_dir> <load 0|1> [out_json]

<tables_dir>: the *_feature_table.csv files for that load, written by
    python3 process_fault.py --all <data_root> <tables_dir> <load>
It must include the healthy table (the temperature trigger is measured
against the healthy motor at the same speed).

Default out_json: trigger_calibration_final_load<load>.json
Re-run whenever you add recordings; with several recordings per fault the
thresholds are fitted to each fault's mean.
"""
import glob
import json
import os
import sys

import pandas as pd

import fault_feature_extraction as F


def load_tables(tables_dir):
    files = sorted(glob.glob(os.path.join(tables_dir, "*_feature_table.csv")))
    if not files:
        raise FileNotFoundError(f"No *_feature_table.csv in {tables_dir}")
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return t[t.fault.isin(["healthy"] + F.V3_FAULTS)]


def add_rise(t):
    hh = {int(r.condition_id): r.to_dict() for _, r in t[t.fault == "healthy"].groupby("condition_id").mean(numeric_only=True).reset_index().iterrows()}
    t = t.copy()
    t["temp__rise_vs_healthy"] = [F.add_relative_features(r.to_dict(), hh.get(int(r.condition_id)))["temp__rise_vs_healthy"]
                                  for _, r in t.iterrows()]
    return t


def main():
    tables_dir, load = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"trigger_calibration_final_load{load}.json"
    t = add_rise(load_tables(tables_dir))
    if "healthy" not in set(t.fault):
        sys.exit("The healthy feature table is needed (temperature rise and Stage A).")
    cal = F.calibrate_triggers(t, definitions=F.definitions_for_load(load))
    for tid, rec in cal.items():
        rec["group"] = F.TRIGGER_GROUP.get(tid)
        rec["tier"] = F.TRIGGER_TIER.get(tid)
    healthy = {int(c): g.mean(numeric_only=True).to_dict() for c, g in t[t.fault == "healthy"].groupby("condition_id")}
    keep = ["accel1__stat_variance", "mic__stat_variance", "temp__temp_mean"]
    payload = {"load": load, "triggers": cal,
               "healthy_reference": {c: {k: v[k] for k in keep} for c, v in healthy.items()}}
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    n = sum(len(r["by_condition"]) for r in cal.values())
    u = sum(x["usable"] for r in cal.values() for x in r["by_condition"].values())
    print(f"load {load}: {len(cal)} triggers, {u} of {n} trigger/speed rules usable -> {out}")


if __name__ == "__main__":
    main()
