"""
rule_consistency.py
====================
For a list of candidate triggers and ONE load: fit each rule on the full
10 s runs, then check it on independent looks at the same speed:
    constant speeds: both 5 s halves; ramps: the reverse-direction ramp.
A rule is consistent at a speed if it fires for its own fault and for none
of the other five faults in every look.
Envelope (ramp-shape) features depend on ramp direction and are marked
"not checkable" for ramps.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import fault_feature_extraction as F

warnings.filterwarnings("ignore")
SWAP = {5: 7, 7: 5, 6: 8, 8: 6}
F6 = ["bowed_rotor", "faulty_bearing", "broken_rotor_bars", "rotor_misalignment", "rotor_unbalance", "stator_winding"]


def add_rise(t, key=("condition_id",)):
    """temperature rise vs the healthy motor at the same speed (and half)"""
    t = t.copy()
    keys = list(key)
    hh = t[t.fault == "healthy"].set_index(keys)["temp__temp_mean"]
    t["temp__rise_vs_healthy"] = [r.temp__temp_mean - hh.loc[tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]]
                                  for _, r in t.iterrows()]
    return t


def load(load_dir):
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(load_dir, "*_feature_table.csv"))], ignore_index=True)
    h = pd.read_csv(os.path.join(load_dir, "halves_features.csv"))
    return add_rise(t), add_rise(h, ("condition_id", "half"))


def check(load_dir, defs):
    t, h = load(load_dir)
    t6, h6 = t[t.fault.isin(F6 + ["healthy"])], h[h.fault.isin(F6 + ["healthy"])]
    cal = F.calibrate_triggers(t6, definitions=defs)
    rows = []
    for tid, rec in cal.items():
        for c, rule in rec["by_condition"].items():
            if not rule["usable"]:
                rows.append(dict(trigger=tid, fault=rec["fault"], cond=c, usable=False, consistent=False, note="not separable"))
                continue
            if c >= 5 and "__env_" in rec["feature"]:
                rows.append(dict(trigger=tid, fault=rec["fault"], cond=c, usable=True, consistent=None, note="not checkable (direction)"))
                continue
            looks = [h6[(h6.half == x) & (h6.condition_id == c)] for x in ("A", "B")] if c <= 4 else [t6[t6.condition_id == SWAP[c]]]
            ok, fa = True, 0
            for L in looks:
                L = L[L.fault.isin(F6)]
                hit = {r.fault: F._check(rule, r[rec["feature"]]) for _, r in L.iterrows()}
                ok &= hit.get(rec["fault"], 0) == 1
                fa += sum(v for k, v in hit.items() if k != rec["fault"])
            rows.append(dict(trigger=tid, fault=rec["fault"], cond=c, usable=True, consistent=bool(ok and fa == 0),
                             note=f"false alarms {fa}" if fa else ""))
    return pd.DataFrame(rows), cal
