"""
validate_final.py
==================
Held-out check of the final, load-aware trigger set
(fault_feature_extraction.definitions_for_load).

    python3 validate_final.py <unloaded_tables_dir> <loaded_tables_dir>

Each tables dir needs the *_feature_table.csv files (process_fault.py) and
halves_features.csv (5 s halves of the constant-speed runs; written by
validate_triggers.py --halves).

Per load, thresholds and the healthy reference are fitted on data the test
never sees:
  constant speeds: fit on one 5 s half of every run, test on the other half
  ramps:           fit on the reverse-direction ramp, test on the ramp
Then each motor's 8 runs are combined (points summed) for one verdict.
"""
import glob
import os
import sys
import warnings

import pandas as pd

import fault_feature_extraction as F

warnings.filterwarnings("ignore")
SWAP = {5: 7, 7: 5, 6: 8, 8: 6}
KEEP = ["healthy"] + F.V3_FAULTS
AB = {"healthy": "HH", "bowed_rotor": "BR", "faulty_bearing": "FB", "broken_rotor_bars": "KA",
      "rotor_misalignment": "RM", "rotor_unbalance": "RU", "stator_winding": "SW"}


def with_rise(t, keys):
    t = t.copy()
    hh = {tuple(r[k] for k in keys): r.to_dict() for _, r in t[t.fault == "healthy"].iterrows()}
    t["temp__rise_vs_healthy"] = [F.add_relative_features(r.to_dict(), hh.get(tuple(r[k] for k in keys)))["temp__rise_vs_healthy"]
                                  for _, r in t.iterrows()]
    return t


def score(test, cal, href):
    rows = []
    for _, r in test.iterrows():
        v, c = r.to_dict(), int(r.condition_id)
        faulty, _ = F.is_faulty(v, c, href.get(c))
        s, fired, best = F.score_triggers(v, c, cal)
        rows.append({"fault": r.fault, "cond": c, "faulty": faulty,
                     "pred": "healthy" if faulty is False else (best or "none"),
                     "pts": {k: x["points"] for k, x in s.items()}})
    return pd.DataFrame(rows)


def verdicts(res):
    out = {}
    for f, g in res.groupby("fault"):
        if (g.faulty == False).all():  # noqa: E712
            out[f] = "healthy"
            continue
        tot = {}
        for p in g.pts:
            for k, x in p.items():
                tot[k] = tot.get(k, 0) + x
        ranked = sorted(tot.items(), key=lambda kv: -kv[1])
        out[f] = (ranked[0][0] if ranked and ranked[0][1] > 0 else "none",
                  ranked[0][1] if ranked else 0, ranked[1][1] if len(ranked) > 1 else 0)
    return out


def run_load(d, load):
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*_feature_table.csv"))], ignore_index=True)
    h = pd.read_csv(os.path.join(d, "halves_features.csv"))
    t, h = with_rise(t[t.fault.isin(KEEP)], ["condition_id"]), with_rise(h[h.fault.isin(KEEP)], ["condition_id", "half"])
    defs = F.definitions_for_load(load)
    defs_r = [x for x in defs if "__env_" not in x[2]]
    ramps = t[t.condition_id.isin(SWAP)]
    fit_r = ramps.copy()
    fit_r["condition_id"] = fit_r.condition_id.map(SWAP)
    rr = score(ramps, F.calibrate_triggers(fit_r, definitions=defs_r),
               {int(r.condition_id): r.to_dict() for _, r in fit_r[fit_r.fault == "healthy"].iterrows()})
    print(f"\n######## {'UNLOADED' if load == 0 else 'LOADED'}  ({len(defs)} triggers: both + {'unloaded' if load == 0 else 'loaded'}-only)")
    for fh, th in (("A", "B"), ("B", "A")):
        fit = h[h.half == fh]
        href = {int(r.condition_id): r.to_dict() for _, r in fit[fit.fault == "healthy"].iterrows()}
        rc = score(h[h.half == th], F.calibrate_triggers(fit, definitions=defs), href)
        res = pd.concat([rc, rr])
        v = verdicts(res)
        acc = (res.fault == res.pred).mean()
        ok = sum((x if isinstance(x, str) else x[0]) == f for f, x in v.items())
        print(f"fit half {fh} / test half {th} + reverse ramps: per-run {acc:.0%}, motors correct {ok}/{len(v)}")
        for f, x in v.items():
            if isinstance(x, str):
                print(f"   {AB[f]:3s} -> {x}")
            else:
                print(f"   {AB[f]:3s} -> {AB.get(x[0], x[0]):4s} {x[1]:5.2f} pts (runner-up {x[2]:.2f})")


if __name__ == "__main__":
    run_load(sys.argv[1], 0)
    run_load(sys.argv[2], 1)
