"""
predict_all.py
===============
Gives THREE diagnoses of one motor, side by side:

    1. triggers  - the hand-found triggers only      (predict_motor.py)
    2. ML        - the ML-found triggers only        (ml/predict_ml.py)
    3. combined  - the points of 1 and 2 added together

    python3 predict_all.py <folder_or_csv_files...> --load 0|1 [--speeds speeds.csv]
                           [--healthy <folder>] [--details] [--json out.json]

The two trigger sets stay separate: each keeps its own list and its own
calibration file, and this script only reads their results. The features of
each run are computed once and handed to both.

All three use the same Stage A (is the motor faulty at all?). The fault type
is classified only when at least one run is faulty; otherwise all three
verdicts are "healthy". Verdict: the fault with the most points; "clear" if
it has >= 3 points and >= 2x the runner-up, else "weak".

--details also prints which triggers fired in every run.
"""
import argparse
import glob
import json
import os
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(HERE, "ml"), os.path.join(os.path.dirname(HERE), "ml_discovery")):
    if os.path.isdir(d):
        sys.path.insert(0, d)
        break
import predict_motor as PM        # noqa: E402  hand-found triggers
import predict_ml as PML          # noqa: E402  ML-found triggers
from process_fault import SPEED_LABELS  # noqa: E402

warnings.filterwarnings("ignore")


def diagnose_all(files, load, speeds=None, healthy_folder=None, verbose=True, details=False):
    vectors = {}
    for f in files:
        cid = PM.speed_of(f, speeds)
        if cid not in SPEED_LABELS:
            raise ValueError(f"{f}: unknown speed profile (use X_X_<1-8>_<load>.csv names or --speeds)")
        vectors[f] = PM.run_features(f, cid)
    hand = PM.diagnose(files, load, speeds, healthy_folder, verbose=False, vectors=vectors)
    ml = PML.diagnose_ml(files, load, speeds, healthy_folder, verbose=False, vectors=vectors)
    faulty = hand["faulty_runs"] > 0
    pts = {k: hand["points"].get(k, 0.0) + ml["points"].get(k, 0.0) for k in set(hand["points"]) | set(ml["points"])}
    pts = dict(sorted(pts.items(), key=lambda kv: -kv[1]))
    v, c = PML.verdict_from_points(pts, faulty)
    combined = {"verdict": v, "confidence": c, "points": pts}
    result = {"load": load, "n_runs": len(files), "faulty_runs": hand["faulty_runs"], "classified": faulty,
              "triggers": {k: hand[k] for k in ("verdict", "confidence", "points")},
              "ml": {k: ml[k] for k in ("verdict", "confidence", "points")},
              "combined": combined,
              "runs": [{"file": h["file"], "condition_id": h["condition_id"], "faulty": h["faulty"],
                        "stage_a": h["stage_a"], "fired_triggers": h["fired"], "fired_ml": m["fired"]}
                       for h, m in zip(hand["runs"], ml["runs"])],
              "notes": hand["notes"]}
    if verbose:
        report(result, details)
    return result


def _line(name, r, faulty):
    p = ", ".join(f"{k} {x:.2f}" for k, x in r["points"].items() if x > 0) if faulty else "not classified"
    return f"  {name:10s} {r['verdict']:28s} {r['confidence']:6s}  points: {p or 'none'}"


def report(r, details=False):
    print(f"\nMotor diagnosis ({'loaded' if r['load'] else 'unloaded'}, {r['n_runs']} runs)")
    print("\n  Stage A (is it faulty?)")
    for run in sorted(r["runs"], key=lambda x: x["condition_id"]):
        a = ", ".join(f"{k.split('__')[0]} {v:,.1f}x" for k, v in run["stage_a"].items())
        print(f"    {run['file']:14s} [{SPEED_LABELS[run['condition_id']]:20s}] "
              f"{'FAULTY' if run['faulty'] else 'healthy level':13s} ({a} healthy)")
        if details and r["classified"]:
            for t in run["fired_triggers"]:
                print(f"        trigger  {t['trigger']:14s} -> {t['fault']}")
            for t in run["fired_ml"]:
                print(f"        ML       {t['trigger']:14s} -> {t['fault']}  ({t['meaning']})")
    print(f"\n  {r['faulty_runs']}/{r['n_runs']} runs above the healthy level")
    print("\n  THREE VERDICTS")
    print(_line("triggers", r["triggers"], r["classified"]))
    print(_line("ML", r["ml"], r["classified"]))
    print(_line("combined", r["combined"], r["classified"]))
    agree = len({r["triggers"]["verdict"], r["ml"]["verdict"]}) == 1
    print(f"\n  triggers and ML {'agree' if agree else 'DISAGREE - check this motor more closely'}")
    for n in r["notes"]:
        print(f"  note: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--load", type=int, choices=[0, 1], required=True)
    ap.add_argument("--speeds")
    ap.add_argument("--healthy")
    ap.add_argument("--details", action="store_true")
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
    res = diagnose_all(files, a.load, speeds, a.healthy, details=a.details)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"\n  saved {a.json}")


if __name__ == "__main__":
    main()
