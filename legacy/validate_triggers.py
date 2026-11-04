"""
validate_triggers.py
=====================
Checks how well the trigger engine identifies each fault.

    python3 validate_triggers.py <tables_dir>                       # in-sample
    python3 validate_triggers.py <tables_dir> --halves <data_root>  # + held-out test

In-sample: thresholds are fitted on all runs, then every run is scored.
This only shows the triggers are consistent with the data they came from
-- it will look optimistic.

Held-out (--halves): constant-speed runs (conditions 1-4) are split into
their first and last 5 s. Thresholds are fitted on one half of every run
and tested on the other half, then the other way round. The test half was
never seen during calibration, so this is the honest number. Ramps cannot
be split this way (each half covers a different speed range).

Cross-ramp (--cross-ramp): 15->45 and 45->15 cover the same speeds in
two separate recordings, as do 30->60 and 60->30. Thresholds for each
ramp are fitted on the reverse ramp and tested on the forward one (and
vice versa). Envelope-shape triggers are left out of this test because
they depend on the ramp direction.

Healthy runs (healthy_feature_table.csv in tables_dir) are scored too:
the correct answer for them is "healthy" from Stage A.
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

from fault_feature_extraction import (build_feature_vector, calibrate_triggers,
                                      score_triggers, is_faulty)
from process_fault import load_data, TIME_CSV, CHANNEL_MAP, FAULT_LABELS, FALLBACK_FS

warnings.filterwarnings("ignore")
ABBR = {v: k for k, v in FAULT_LABELS.items()}


def healthy_refs(table):
    h = table[table.fault == "healthy"]
    return {int(r["condition_id"]): r.to_dict() for _, r in h.iterrows()}


def evaluate(table, cal, href=None):
    rows = []
    href = href or {}
    for _, r in table.iterrows():
        vec = r.to_dict()
        c = int(r["condition_id"])
        faulty, _ = is_faulty(vec, c, href.get(c))
        scores, fired, best = score_triggers(vec, c, cal)
        if faulty is False:
            best = "healthy"
        true = r["fault"]
        pts = sorted(((s["points"], f) for f, s in scores.items()), reverse=True)
        tie = len(pts) > 1 and pts[0][0] == pts[1][0] and pts[0][0] > 0
        rows.append({"fault": ABBR.get(true, true), "cond": c,
                     "pred": ABBR.get(best, best) if best else "none",
                     "correct": best == true, "tie": tie, "faulty_flag": faulty,
                     "true_pts": scores.get(true, {}).get("points", 0.0),
                     "true_possible": scores.get(true, {}).get("possible", 0.0)})
    return pd.DataFrame(rows)


def summarise(res, title):
    print(f"\n=== {title} ===")
    print(f"Accuracy: {res.correct.mean():.0%} ({res.correct.sum()}/{len(res)})"
          f"   stage A flagged faulty: {res.faulty_flag.mean():.0%}")
    grid = res.pivot(index="fault", columns="cond", values="pred")
    ok = res.pivot(index="fault", columns="cond", values="correct")
    shown = grid.copy().astype(object)
    for i in grid.index:
        for j in grid.columns:
            shown.loc[i, j] = grid.loc[i, j] if ok.loc[i, j] else f"{grid.loc[i, j]}*"
    print("Predicted fault per run (* = wrong):")
    print(shown.to_string())
    print("Accuracy by fault:", res.groupby("fault").correct.mean().round(2).to_dict())
    print("Accuracy by condition:", res.groupby("cond").correct.mean().round(2).to_dict())


def half_tables(data_root, load=0):
    rows = []
    for code, label in FAULT_LABELS.items():
        for f in sorted(glob.glob(os.path.join(data_root, code, f"{code[0]}_{code[1]}_*_{load}.csv"))):
            c = int(os.path.basename(f).split("_")[2])
            if c > 4:
                continue
            df = load_data(TIME_CSV, f)
            n = len(df) // 2
            for h, part in (("A", df.iloc[:n]), ("B", df.iloc[n:])):
                part = part.reset_index(drop=True)
                vec = build_feature_vector(part, FALLBACK_FS, channel_map=CHANNEL_MAP, condition_id=c)
                vec.update({"fault": label, "condition_id": c, "half": h})
                rows.append(vec)
            print(f"  halves done: {os.path.basename(f)}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables_dir")
    ap.add_argument("--halves", metavar="DATA_ROOT")
    ap.add_argument("--cross-ramp", action="store_true")
    ap.add_argument("--load", type=int, default=0, help="0 = unloaded files (_0), 1 = loaded (_1)")
    ap.add_argument("--save", default="validation_results.csv")
    a = ap.parse_args()

    ft = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.tables_dir, "*_feature_table.csv"))],
                   ignore_index=True)
    cal = calibrate_triggers(ft)
    res = evaluate(ft, cal, healthy_refs(ft))
    res["test"] = "in-sample"
    summarise(res, f"In-sample (fit and test on the same {len(ft)} runs)")
    out = [res]

    if a.halves:
        cache = os.path.join(a.tables_dir, "halves_features.csv")
        if os.path.exists(cache):
            ht = pd.read_csv(cache)
        else:
            print("\nComputing features on 5 s halves of the constant-speed runs ...")
            ht = half_tables(a.halves, a.load)
            ht.to_csv(cache, index=False)
        parts = []
        for fit_h, test_h in (("A", "B"), ("B", "A")):
            fit = ht[ht.half == fit_h]
            cal_h = calibrate_triggers(fit)
            r = evaluate(ht[ht.half == test_h], cal_h, healthy_refs(fit))
            r["test"] = f"fit {fit_h} -> test {test_h}"
            parts.append(r)
            summarise(r, f"Held-out, constant speeds: fit on half {fit_h}, test on half {test_h}")
        both = pd.concat(parts)
        print(f"\nHeld-out accuracy, both directions: {both.correct.mean():.0%} ({both.correct.sum()}/{len(both)})")
        out += parts

    if a.cross_ramp:
        from fault_feature_extraction import TRIGGER_DEFINITIONS
        defs = [d for d in TRIGGER_DEFINITIONS if "__env_" not in d[2]]
        swap = {5: 7, 7: 5, 6: 8, 8: 6}
        ramps = ft[ft.condition_id.isin(swap)].copy()
        fit = ramps.copy()
        fit["condition_id"] = fit["condition_id"].map(swap)   # reverse ramp stands in for the forward one
        cal_x = calibrate_triggers(fit, definitions=defs)
        r = evaluate(ramps, cal_x, healthy_refs(fit))
        r["test"] = "cross-ramp"
        summarise(r, "Held-out ramps: thresholds fitted on the reverse-direction recording")
        out.append(r)

    pd.concat(out).to_csv(a.save, index=False)
    print(f"\nSaved per-run results to {a.save}")


if __name__ == "__main__":
    main()
