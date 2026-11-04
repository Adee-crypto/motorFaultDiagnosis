"""
validate_ml.py
===============
Held-out identification check, same protocol as validate_final.py, for
three systems side by side:
    hand : the hand-found triggers only (both-load + this load's group)
    ml   : the ML-found triggers only (ml_triggers_load<L>.json)
    both : hand + ML points added together

    python3 validate_ml.py <unloaded_tables> <loaded_tables> <ml_out_dir>

Thresholds and the healthy reference are fitted on data the test never sees:
    constant speeds: fit on one 5 s half, test on the other half
    ramps:           fit on the reverse-direction ramp, test on the ramp
The ML triggers keep the feature, direction and speeds the search chose;
only their thresholds are re-fitted here. Each motor's 8 runs are combined
(points summed) for one verdict, as in predict_motor.py.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import ml_data as D
import ml_features as M

F = D.F
warnings.filterwarnings("ignore")


def with_derived(tab, ref, names):
    """derived features of every row, relative to ref[condition_id] (healthy raw vector)"""
    rows = [M.derive(r.to_dict(), ref.get(int(r.condition_id)), names) for _, r in tab.iterrows()]
    out = pd.concat([tab.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    hh = {c: v for c, v in ref.items()}
    out["temp__rise_vs_healthy"] = [F.add_relative_features(r.to_dict(), hh.get(int(r.condition_id)))["temp__rise_vs_healthy"]
                                    for _, r in tab.iterrows()]
    return out


def refit_ml(trigs, fit):
    """same feature / direction / speeds, thresholds re-fitted on fit"""
    cal = {}
    f6 = fit[fit.fault.isin(D.F6)]
    for tid, t in trigs.items():
        feat, fault = t["feature"], t["fault"]
        speeds = sorted(int(c) for c in t["by_condition"])
        high = next(iter(t["by_condition"].values()))["op"] == ">="
        by = {}
        vals = {c: f6[f6.condition_id == c].set_index("fault")[feat].dropna() for c in speeds}
        vals = {c: v for c, v in vals.items() if fault in v.index and len(v) == len(D.F6)}
        if t["kind"] == "P" and vals:
            if high:
                lo = max(v.drop(fault).max() for v in vals.values()); hi = min(v[fault] for v in vals.values())
            else:
                lo = max(v[fault] for v in vals.values()); hi = min(v.drop(fault).min() for v in vals.values())
            if lo < hi:
                thr = (lo + hi) / 2
                by = {c: {"op": ">=" if high else "<=", "lo": thr if high else None, "hi": None if high else thr,
                          "usable": True} for c in vals}
        if not by:                       # per speed (S triggers, or P that no longer pools)
            for c, v in vals.items():
                o = v.drop(fault)
                if high and v[fault] > o.max():
                    by[c] = {"op": ">=", "lo": (v[fault] + o.max()) / 2, "hi": None, "usable": True}
                elif not high and v[fault] < o.min():
                    by[c] = {"op": "<=", "lo": None, "hi": (v[fault] + o.min()) / 2, "usable": True}
        cal[tid] = {"fault": fault, "feature": feat, "weight": t["weight"], "by_condition": by}
    return cal


def score(test, cals, href):
    rows = []
    for _, r in test.iterrows():
        v, c = r.to_dict(), int(r.condition_id)
        faulty, _ = F.is_faulty(v, c, href.get(c))
        pts = {}
        for name, cal in cals.items():
            s, _, _ = F.score_triggers(v, c, cal)
            pts[name] = {k: x["points"] for k, x in s.items()}
        pts["both"] = {k: pts["hand"].get(k, 0) + pts["ml"].get(k, 0) for k in set(pts["hand"]) | set(pts["ml"])}
        rows.append({"fault": r.fault, "cond": c, "faulty": faulty, "pts": pts})
    return pd.DataFrame(rows)


def best(p):
    r = sorted(p.items(), key=lambda kv: -kv[1])
    return (r[0][0] if r and r[0][1] > 0 else "none", r[0][1] if r else 0, r[1][1] if len(r) > 1 else 0)


def summarise(res, system):
    per_run = np.mean([("healthy" if not row.faulty else best(row.pts[system])[0]) == row.fault
                       for _, row in res.iterrows()])
    motors = {}
    for f, g in res.groupby("fault"):
        if (g.faulty == False).all():   # noqa: E712
            motors[f] = ("healthy", 0, 0)
            continue
        tot = {}
        for p in g.pts:
            for k, x in p[system].items():
                tot[k] = tot.get(k, 0) + x
        motors[f] = best(tot)
    return per_run, motors


def run_load(d, load, ml_dir):
    trigs = json.load(open(os.path.join(ml_dir, f"ml_triggers_load{load}.json")))["triggers"]
    names = sorted({t["feature"] for t in trigs.values()})
    t, h = D.read_tables(d)
    defs = F.definitions_for_load(load)
    defs_r = [x for x in defs if "__env_" not in x[2]]
    print(f"\n######## {'UNLOADED' if load == 0 else 'LOADED'}: {len(defs)} hand triggers, {len(trigs)} ML triggers")
    # ramps: fit on the reverse ramp
    ramps = t[t.condition_id.isin(D.SWAP)].copy()
    fit_r = ramps.copy(); fit_r["condition_id"] = fit_r.condition_id.map(D.SWAP)
    ref_r = {int(r.condition_id): r.to_dict() for _, r in fit_r[fit_r.fault == "healthy"].iterrows()}
    fit_r, test_r = with_derived(fit_r, ref_r, names), with_derived(ramps, ref_r, names)
    cal_r = {"hand": F.calibrate_triggers(fit_r, definitions=defs_r), "ml": refit_ml(trigs, fit_r)}
    rr = score(test_r, cal_r, ref_r)
    for fh, th in (("A", "B"), ("B", "A")):
        fit, test = h[h.half == fh], h[h.half == th]
        ref = {int(r.condition_id): r.to_dict() for _, r in fit[fit.fault == "healthy"].iterrows()}
        fit, test = with_derived(fit, ref, names), with_derived(test, ref, names)
        cal = {"hand": F.calibrate_triggers(fit, definitions=defs), "ml": refit_ml(trigs, fit)}
        res = pd.concat([score(test, cal, ref), rr], ignore_index=True)
        print(f"-- fit half {fh} / test half {th} + reverse ramps")
        out = {s: summarise(res, s) for s in ("hand", "ml", "both")}
        print("   " + "   ".join(f"{s}: per-run {out[s][0]:.0%}, motors {sum(v[0] == f for f, v in out[s][1].items())}/7"
                                  for s in out))
        for f in ["healthy"] + D.F6:
            print("   " + f"{D.AB[f]:3s}" + "".join(
                f"   {s:4s} -> {D.AB.get(out[s][1][f][0], out[s][1][f][0]):4s} {out[s][1][f][1]:5.2f} (2nd {out[s][1][f][2]:4.2f})"
                for s in out))


if __name__ == "__main__":
    run_load(sys.argv[1], 0, sys.argv[3])
    run_load(sys.argv[2], 1, sys.argv[3])
