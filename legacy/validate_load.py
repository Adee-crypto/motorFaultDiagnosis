"""
validate_load.py
=================
Checks the v3 trigger set (TRIGGER_DEFINITIONS_V3: 6 faults, no voltage
unbalance, no temperature) on both load conditions.

    python3 validate_load.py <unloaded_tables_dir> <loaded_tables_dir> [--tier confirmed]

Each tables dir holds the *_feature_table.csv files from process_fault.py
and, optionally, halves_features.csv (written by validate_triggers.py
--halves) for the 5 s held-out test.

Tests, per load (thresholds and healthy reference from the SAME load):
  in-sample    fit and test on the same recordings (optimistic)
  halves       constant speeds: fit on one 5 s half, test on the other
  cross-ramp   ramps: fit on the reverse-direction recording
and across loads:
  cross-load   fit thresholds on one load, test on the other (healthy
               reference still from the test load)
For every test, per-run accuracy and the 8-speed combined verdict per
motor (sum of trigger points over that motor's runs) are printed.
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

import fault_feature_extraction as F

warnings.filterwarnings("ignore")
AB = {"healthy": "HH", "bowed_rotor": "BR", "faulty_bearing": "FB", "broken_rotor_bars": "KA",
      "rotor_misalignment": "RM", "rotor_unbalance": "RU", "stator_winding": "SW"}
SWAP = {5: 7, 7: 5, 6: 8, 8: 6}


def load_dir(d):
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*_feature_table.csv"))], ignore_index=True)
    t = t[t.fault.isin(AB)]
    h = os.path.join(d, "halves_features.csv")
    ht = pd.read_csv(h) if os.path.exists(h) else None
    if ht is not None:
        ht = ht[ht.fault.isin(AB)]
    return t, ht


def hrefs(t):
    return {int(r.condition_id): r.to_dict() for _, r in t[t.fault == "healthy"].iterrows()}


def score(test, cal, href):
    rows = []
    for _, r in test.iterrows():
        v, c = r.to_dict(), int(r.condition_id)
        faulty, _ = F.is_faulty(v, c, href.get(c))
        s, _, best = F.score_triggers(v, c, cal)
        pred = "healthy" if faulty is False else (best or "none")
        rows.append({"fault": r.fault, "cond": c, "pred": pred, "faulty": faulty,
                     "pts": {k: x["points"] for k, x in s.items()}})
    return pd.DataFrame(rows)


def combined(res):
    out = {}
    for f, g in res.groupby("fault"):
        if f == "healthy":
            out[f] = "healthy" if (g.faulty == False).all() else "faulty"  # noqa: E712
            continue
        tot = {}
        for p in g.pts:
            for k, x in p.items():
                tot[k] = tot.get(k, 0) + x
        out[f] = max(tot, key=tot.get) if tot and max(tot.values()) > 0 else "none"
    return out


def report(name, res, rows):
    acc = (res.fault == res.pred).mean()
    comb = combined(res)
    n_ok = sum(comb[f] == f for f in comb)
    per_fault = res.assign(ok=res.fault == res.pred).groupby("fault").ok.mean()
    print(f"\n{name}\n  per-run accuracy {acc:.0%} ({(res.fault == res.pred).sum()}/{len(res)});  "
          f"all speeds combined: {n_ok}/{len(comb)} motors correct")
    print("  combined verdicts: " + "  ".join(f"{AB[f]}->{AB.get(p, p)}" for f, p in comb.items()))
    for f, a in per_fault.items():
        rows.append({"test": name, "fault": AB[f], "per_run_acc": a, "combined_ok": comb[f] == f,
                     "combined_pred": AB.get(comb[f], comb[f])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unloaded")
    ap.add_argument("loaded")
    ap.add_argument("--tier", choices=["all", "confirmed"], default="all")
    ap.add_argument("--save", default="validation_load_results.csv")
    a = ap.parse_args()
    defs = [d for d in F.TRIGGER_DEFINITIONS_V3 if a.tier == "all" or F.TRIGGER_TIER[d[0]] == "confirmed"]
    print(f"Trigger set v3, tier = {a.tier}: {len(defs)} triggers")

    data = {0: load_dir(a.unloaded), 1: load_dir(a.loaded)}
    rows = []
    for ld, (t, ht) in data.items():
        tag = "unloaded" if ld == 0 else "loaded"
        cal = F.calibrate_triggers(t, definitions=defs)
        report(f"[{tag}] in-sample", score(t, cal, hrefs(t)), rows)
        if ht is not None:
            parts = []
            for fh, th in (("A", "B"), ("B", "A")):
                fit = ht[ht.half == fh]
                parts.append(score(ht[ht.half == th], F.calibrate_triggers(fit, definitions=defs), hrefs(fit)))
            report(f"[{tag}] held-out 5 s halves (constant speeds, both directions)", pd.concat(parts), rows)
        ramps = t[t.condition_id.isin(SWAP)]
        fit = ramps.copy()
        fit["condition_id"] = fit.condition_id.map(SWAP)
        defs_r = [d for d in defs if "__env_" not in d[2]]
        report(f"[{tag}] held-out ramps (fit on reverse ramp)",
               score(ramps, F.calibrate_triggers(fit, definitions=defs_r), hrefs(fit)), rows)

    for fl, tl in ((0, 1), (1, 0)):
        cal = F.calibrate_triggers(data[fl][0], definitions=defs)
        name = f"cross-load: fit {'unloaded' if fl == 0 else 'loaded'} -> test {'loaded' if tl == 1 else 'unloaded'}"
        report(name, score(data[tl][0], cal, hrefs(data[tl][0])), rows)

    pd.DataFrame(rows).to_csv(a.save, index=False)
    print(f"\nSaved {a.save}")


if __name__ == "__main__":
    main()
