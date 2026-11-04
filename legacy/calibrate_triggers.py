"""
calibrate_triggers.py
======================
Turns the trigger list in fault_feature_extraction.TRIGGER_DEFINITIONS
into numeric thresholds, using labelled feature tables.

Usage:
    python3 calibrate_triggers.py <tables_dir> [out_json] [--set v3|v2]

--set v3 (default): the load-checked trigger set (6 faults, no voltage
unbalance, no temperature). --set v2: the original unloaded-only set.
Calibrate each load separately -- thresholds shift with load:
    python3 calibrate_triggers.py tables_unloaded trigger_calibration_load0.json
    python3 calibrate_triggers.py tables_loaded   trigger_calibration_load1.json

<tables_dir> holds the *_feature_table.csv files written by
    python3 process_fault.py --all <data_root> <tables_dir>

Writes trigger_calibration.json (used by identify_fault.py) and prints a
report: for every trigger and speed condition, the threshold, the fault's
own value, its nearest competitor, and whether the separation is big
enough for the trigger to be used ("usable").

Re-run this whenever you add data (e.g. healthy runs, or repeat recordings
of a fault) -- with several recordings per fault the thresholds are fitted
to the per-fault mean, so they stop depending on a single file.
"""
import glob
import json
import os
import sys

import pandas as pd

from fault_feature_extraction import calibrate_triggers, TRIGGER_DEFINITIONS, TRIGGER_DEFINITIONS_V3, V3_FAULTS


def load_tables(tables_dir):
    files = sorted(glob.glob(os.path.join(tables_dir, "*_feature_table.csv")))
    if not files:
        raise FileNotFoundError(f"No *_feature_table.csv in {tables_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fmt(x):
    if x is None:
        return "-"
    ax = abs(x)
    return f"{x:.3g}" if (ax >= 1e-3 or ax == 0) else f"{x:.2e}"


def report(cal):
    lines = []
    for tid, rec in cal.items():
        lines.append(f"\n{tid:12s} {rec['fault']:20s} {rec['feature']}  (w={rec['weight']})  -- {rec['note']}")
        for c, r in sorted(rec["by_condition"].items(), key=lambda kv: int(kv[0])):
            if r["op"] == ">=":
                rule = f">= {fmt(r['lo'])}"
            elif r["op"] == "<=":
                rule = f"<= {fmt(r['hi'])}"
            else:
                rule = f"in [{fmt(r['lo'])}, {fmt(r['hi'])}]"
            flag = "usable" if r["usable"] else "NOT separable"
            lines.append(f"    cond {c}: {rule:24s} fault={fmt(r['value']):>9s}  nearest={r['neighbour']:<19s} sep={r['separation']:.2f}  {flag}")
    return "\n".join(lines)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    which = "v2" if "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "v2" else "v3"
    args = [x for x in args if x not in ("v2", "v3")]
    tables_dir = args[0] if args else "."
    out_json = args[1] if len(args) > 1 else "trigger_calibration.json"
    ft = load_tables(tables_dir)
    if which == "v3":
        ft = ft[ft.fault.isin(V3_FAULTS + ["healthy"])]
    print(f"Trigger set {which}. Loaded {len(ft)} runs: {ft.groupby('fault').size().to_dict()}")
    cal = calibrate_triggers(ft, definitions=TRIGGER_DEFINITIONS_V3 if which == "v3" else TRIGGER_DEFINITIONS)
    with open(out_json, "w") as fh:
        json.dump(cal, fh, indent=1, default=float)
    print(report(cal))
    n_rules = sum(len(r["by_condition"]) for r in cal.values())
    n_use = sum(r2["usable"] for r in cal.values() for r2 in r["by_condition"].values())
    print(f"\n{n_use} of {n_rules} trigger/condition rules usable. Saved {out_json}")

    # coverage: usable triggers per fault per condition
    cov = {}
    for rec in cal.values():
        for c, r in rec["by_condition"].items():
            if r["usable"]:
                cov.setdefault(rec["fault"], {}).setdefault(int(c), 0)
                cov[rec["fault"]][int(c)] += 1
    print("\nUsable triggers per fault and condition:")
    print(pd.DataFrame(cov).T.reindex(columns=range(1, 9)).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
