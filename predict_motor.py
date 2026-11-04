"""
predict_motor.py
=================
Diagnoses ONE motor from its runs (ideally all 8 speed profiles), given
whether it was loaded and the speed profile of each run.

    python3 predict_motor.py <folder_or_csv_files...> --load 0|1 [--speeds speeds.csv]
                             [--healthy <folder of healthy runs>] [--cal <json>] [--json out.json]

Inputs
  runs      a folder with the motor's CSV files, or the files themselves.
            Each CSV has the sensor columns (Accelerometer 1/2/3, Microphone,
            Temperature), 420,000 rows = 10 s at 41,666.7 Hz.
  --load    0 = unloaded, 1 = loaded (required; thresholds differ per load)
  speed     per run, the condition id 1-8:
              1: 15 Hz, 2: 30 Hz, 3: 45 Hz, 4: 60 Hz (constant)
              5: 15->45, 6: 30->60, 7: 45->15, 8: 60->30 Hz (ramps)
            read from file names like X_X_5_1.csv, or given in --speeds, a
            CSV with columns file,condition_id
  --healthy optional folder with healthy runs of the same rig at the same
            load: used as the healthy reference instead of the built-in one
  --cal     calibration file (default trigger_calibration_final_load<load>.json
            next to this script)

This file uses the hand-found triggers only. For the ML-found triggers see
ml/predict_ml.py, and for all three verdicts side by side predict_all.py.

Output
  printed report: per run, Stage A (faulty?) and, if the motor is faulty,
  the triggers that fired; then points per fault summed over all runs, the
  verdict and how clear it is. --json writes the same as a machine-readable
  file.

Verdict rules
  healthy  : no run is above 2x the healthy vibration/sound level; the
             fault-type triggers (Stage B) are then not checked at all,
             because they separate faults from each other, not from healthy
  <fault>  : the fault with the most trigger points over all runs
  confidence "clear" if the winner has >= 3 points and >= 2x the runner-up,
             else "weak"; "unknown fault type" if no trigger fired
Voltage unbalance is not detectable with these sensors and is never
predicted; such a motor would show up as faulty with a weak or wrong type.
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings

import pandas as pd

import fault_feature_extraction as F
from process_fault import load_data, estimate_sample_rate, TIME_CSV, CHANNEL_MAP, SPEED_LABELS

warnings.filterwarnings("ignore")


def read_calibration(path):
    with open(path) as fh:
        cal = json.load(fh)
    for rec in cal["triggers"].values():
        rec["by_condition"] = {int(k): v for k, v in rec["by_condition"].items()}
    cal["healthy_reference"] = {int(k): v for k, v in cal["healthy_reference"].items()}
    return cal


def run_features(path, cid):
    df = load_data(TIME_CSV, path)
    fs = estimate_sample_rate(df["Time"].values)
    return F.build_feature_vector(df, fs, channel_map=CHANNEL_MAP, condition_id=cid)


def speed_of(path, table):
    if table is not None and os.path.basename(path) in table:
        return table[os.path.basename(path)]
    m = re.search(r"_(\d)_\d\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def healthy_refs(folder, load, cal):
    """per-speed healthy reference: user's healthy runs, else the calibration's"""
    if not folder:
        return cal["healthy_reference"]
    refs = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        cid = speed_of(f, None)
        if cid:
            refs[cid] = run_features(f, cid)
    missing = sorted(set(range(1, 9)) - set(refs))
    for c in missing:
        refs[c] = cal["healthy_reference"].get(c)
    return refs


HERE = os.path.dirname(os.path.abspath(__file__))


def diagnose(files, load, speeds=None, healthy_folder=None, cal_path=None, verbose=True, vectors=None):
    """vectors: optional {file path: raw feature vector}, so a caller that has
    already computed the features (predict_all.py) does not compute them twice"""
    cal = read_calibration(cal_path or os.path.join(HERE, f"trigger_calibration_final_load{load}.json"))
    if cal["load"] != load:
        raise ValueError(f"calibration file is for load {cal['load']}, not {load}")
    href = healthy_refs(healthy_folder, load, cal)

    # Stage A on every run: is the motor faulty at all?
    runs, vecs = [], []
    for f in files:
        cid = speed_of(f, speeds)
        if cid not in SPEED_LABELS:
            raise ValueError(f"{f}: unknown speed profile (use X_X_<1-8>_<load>.csv names or --speeds)")
        raw = vectors[f] if vectors and f in vectors else run_features(f, cid)
        vec = F.add_relative_features(raw, href.get(cid))
        faulty, evA = F.is_faulty(vec, cid, href.get(cid))
        vecs.append(vec)
        runs.append({"file": os.path.basename(f), "path": f, "condition_id": cid, "speed": SPEED_LABELS[cid],
                     "faulty": bool(faulty), "stage_a": evA, "fired": []})
    n_faulty = sum(r["faulty"] for r in runs)

    # Stage B only if Stage A found a fault: the triggers separate faults
    # from each other, not faults from healthy, so on a healthy motor they
    # would fire meaninglessly.
    totals, classified = {}, n_faulty > 0
    if classified:
        for run, vec in zip(runs, vecs):
            scores, fired, _ = F.score_triggers(vec, run["condition_id"], cal["triggers"])
            for k, sc in scores.items():
                totals[k] = totals.get(k, 0.0) + sc["points"]
            run["fired"] = [{"trigger": t, "fault": fa, "group": F.TRIGGER_GROUP.get(t), "feature": ft, "value": x}
                            for t, fa, ft, x, rule, hit in fired if hit]
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    if not classified:
        verdict, conf = "healthy", "clear"
    elif not ranked or ranked[0][1] == 0:
        verdict, conf = "faulty - unknown fault type", "weak"
    else:
        verdict = ranked[0][0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        conf = "clear" if ranked[0][1] >= 3 and ranked[0][1] >= 2 * second else "weak"
    notes = ["fault type is classified only when the motor is faulty (at least one run above the healthy "
             "level in Stage A); " + ("this motor is faulty, so the triggers were checked" if classified
                                      else "no run is faulty, so the fault-type triggers were not checked")]
    if len(runs) < 8:
        notes.append(f"only {len(runs)} of 8 speed profiles given; single runs are much less reliable")
    result = {"load": load, "runs": runs, "points": dict(ranked), "verdict": verdict, "confidence": conf,
              "faulty_runs": n_faulty, "classified": classified, "notes": notes}
    if verbose:
        report(result)
    return result


def report(r):
    print(f"\nMotor diagnosis ({'loaded' if r['load'] else 'unloaded'}, {len(r['runs'])} runs)")
    for run in sorted(r["runs"], key=lambda x: x["condition_id"]):
        a = ", ".join(f"{k.split('__')[0]} {v:,.1f}x" for k, v in run["stage_a"].items())
        print(f"\n  {run['file']}  [{run['speed']}]  {'FAULTY' if run['faulty'] else 'healthy level'}  ({a} healthy)")
        for t in run["fired"]:
            print(f"     fired: {t['trigger']:14s} {t['group']:8s} -> {t['fault']}")
    print(f"\n  Stage A: {r['faulty_runs']}/{len(r['runs'])} runs above the healthy level")
    if r["classified"]:
        print("  Points:", ", ".join(f"{k} {v:.2f}" for k, v in r["points"].items() if v > 0) or "none")
    else:
        print("  Stage B (fault type): skipped, the motor is at the healthy level")
    print(f"\n  VERDICT: {r['verdict']}   (confidence: {r['confidence']})")
    for n in r["notes"]:
        print(f"  note: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--load", type=int, choices=[0, 1], required=True)
    ap.add_argument("--speeds", help="CSV with columns file,condition_id")
    ap.add_argument("--healthy", help="folder with healthy runs at the same load")
    ap.add_argument("--cal")
    ap.add_argument("--json")
    a = ap.parse_args()
    files = []
    for p in a.runs:
        files += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("no CSV files found")
    speeds = None
    if a.speeds:
        s = pd.read_csv(a.speeds)
        speeds = dict(zip(s["file"].map(os.path.basename), s["condition_id"].astype(int)))
    res = diagnose(files, a.load, speeds, a.healthy, a.cal)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"\n  saved {a.json}")


if __name__ == "__main__":
    main()
