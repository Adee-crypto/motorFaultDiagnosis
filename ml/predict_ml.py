"""
predict_ml.py
==============
Diagnoses ONE motor using the ML-found triggers only (ml_triggers_load<L>.json,
made by discover.py and prune_ml.py). Needs numpy/scipy/pandas/PyWavelets,
NOT scikit-learn: the ML was only used to find the triggers.

    python3 ml/predict_ml.py <folder_or_csv_files...> --load 0|1 [--speeds speeds.csv]
                             [--healthy <folder>] [--cal <json>] [--json out.json]

Same inputs, Stage A and verdict rules as predict_motor.py:
  Stage A  a run is faulty if accelerometer-1 or microphone variance is at
           least 2x the healthy motor at the same speed and load
  Stage B  only if a run is faulty: each ML trigger that fires adds its weight
           (0.5) to its fault, summed over all runs
  verdict  the fault with the most points; "clear" if >= 3 points and >= 2x
           the runner-up, else "weak"

The ML triggers use features relative to the healthy motor (rel__) and
ratios between two sensors (xch__), see ml_features.py.
"""
import argparse
import glob
import json
import os
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ml_data as D                    # noqa: E402  (also puts the trigger code on the path)
import ml_features as M                # noqa: E402
from process_fault import load_data, estimate_sample_rate, TIME_CSV, CHANNEL_MAP, SPEED_LABELS  # noqa: E402

F = D.F
warnings.filterwarnings("ignore")


def read_ml_calibration(path):
    with open(path) as fh:
        cal = json.load(fh)
    for t in cal["triggers"].values():
        t["by_condition"] = {int(k): v for k, v in t["by_condition"].items()}
    cal["healthy_reference"] = {int(k): v for k, v in cal["healthy_reference"].items()}
    return cal


def run_features(path, cid):
    df = load_data(TIME_CSV, path)
    fs = estimate_sample_rate(df["Time"].values)
    return F.build_feature_vector(df, fs, channel_map=CHANNEL_MAP, condition_id=cid)


def verdict_from_points(points, any_faulty):
    ranked = sorted(points.items(), key=lambda kv: -kv[1])
    if not any_faulty:
        return "healthy", "clear"
    if not ranked or ranked[0][1] == 0:
        return "faulty - unknown fault type", "weak"
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], ("clear" if ranked[0][1] >= 3 and ranked[0][1] >= 2 * second else "weak")


def score_runs(runs, cal, healthy=None):
    """runs: list of {"file", "condition_id", "raw": raw feature vector}.
    healthy: optional {condition_id: raw healthy vector}; default = the
    reference stored in the ML calibration file.
    Returns (per-run results, points per fault, number of faulty runs)."""
    href = healthy or cal["healthy_reference"]
    names = sorted({t["feature"] for t in cal["triggers"].values()})
    out, n_faulty = [], 0
    for r in runs:
        h = href.get(r["condition_id"])
        faulty, evA = F.is_faulty(r["raw"], r["condition_id"], h)
        n_faulty += bool(faulty)
        out.append({"file": r["file"], "condition_id": r["condition_id"], "faulty": bool(faulty),
                    "stage_a": evA, "vec": dict(r["raw"], **M.derive(r["raw"], h, names)), "fired": []})
    totals = {}
    if n_faulty:
        for r in out:
            scores, fired, _ = F.score_triggers(r["vec"], r["condition_id"], cal["triggers"])
            for k, s in scores.items():
                totals[k] = totals.get(k, 0.0) + s["points"]
            r["fired"] = [{"trigger": t, "fault": fa, "feature": ft, "meaning": M.describe(ft), "value": x}
                          for t, fa, ft, x, rule, hit in fired if hit]
    for r in out:
        r.pop("vec")
    return out, dict(sorted(totals.items(), key=lambda kv: -kv[1])), n_faulty


def diagnose_ml(files, load, speeds=None, healthy_folder=None, cal_path=None, verbose=True, vectors=None):
    cal = read_ml_calibration(cal_path or os.path.join(HERE, f"ml_triggers_load{load}.json"))
    if cal["load"] != load:
        raise ValueError(f"ML calibration file is for load {cal['load']}, not {load}")
    runs = []
    for f in files:
        cid = speed_of(f, speeds)
        if cid not in SPEED_LABELS:
            raise ValueError(f"{f}: unknown speed profile (use X_X_<1-8>_<load>.csv names or --speeds)")
        raw = vectors[f] if vectors and f in vectors else run_features(f, cid)
        runs.append({"file": os.path.basename(f), "condition_id": cid, "raw": raw})
    healthy = None
    if healthy_folder:
        healthy = dict(cal["healthy_reference"])
        for f in sorted(glob.glob(os.path.join(healthy_folder, "*.csv"))):
            c = speed_of(f, None)
            if c:
                healthy[c] = run_features(f, c)
    res, points, n_faulty = score_runs(runs, cal, healthy)
    verdict, conf = verdict_from_points(points, n_faulty > 0)
    result = {"load": load, "runs": res, "points": points, "verdict": verdict, "confidence": conf,
              "faulty_runs": n_faulty, "classified": n_faulty > 0}
    if verbose:
        report_ml(result)
    return result


def speed_of(path, table):
    import re
    if table is not None and os.path.basename(path) in table:
        return table[os.path.basename(path)]
    m = re.search(r"_(\d)_\d\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def report_ml(r):
    print(f"\nML-trigger diagnosis ({'loaded' if r['load'] else 'unloaded'}, {len(r['runs'])} runs)")
    for run in sorted(r["runs"], key=lambda x: x["condition_id"]):
        print(f"\n  {run['file']}  [{SPEED_LABELS[run['condition_id']]}]  {'FAULTY' if run['faulty'] else 'healthy level'}")
        for t in run["fired"]:
            print(f"     fired: {t['trigger']:13s} -> {t['fault']:20s} ({t['meaning']})")
    print(f"\n  Stage A: {r['faulty_runs']}/{len(r['runs'])} runs above the healthy level")
    if r["classified"]:
        print("  ML points:", ", ".join(f"{k} {v:.2f}" for k, v in r["points"].items() if v > 0) or "none")
    else:
        print("  Stage B (fault type): skipped, the motor is at the healthy level")
    print(f"\n  ML VERDICT: {r['verdict']}   (confidence: {r['confidence']})")
    print("  note: fault type is classified only when the motor is faulty (Stage A)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--load", type=int, choices=[0, 1], required=True)
    ap.add_argument("--speeds")
    ap.add_argument("--healthy")
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
    res = diagnose_ml(files, a.load, speeds, a.healthy, a.cal)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
