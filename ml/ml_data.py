"""
ml_data.py
===========
Loads the feature tables of one load and adds the ML features
(ml_features.py), relative to the healthy motor at the same speed and load
(and the same 5 s half for the half-recordings).

Also lists the (fault, feature) pairs the hand-found triggers already use,
so the ML search can stay away from them.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))   # dev layout
sys.path.insert(0, os.path.dirname(HERE))                         # package layout
import fault_feature_extraction as F   # noqa: E402
import ml_features as M                # noqa: E402

F6 = ["bowed_rotor", "faulty_bearing", "broken_rotor_bars", "rotor_misalignment", "rotor_unbalance", "stator_winding"]
AB = {"bowed_rotor": "BR", "faulty_bearing": "FB", "broken_rotor_bars": "KA", "rotor_misalignment": "RM",
      "rotor_unbalance": "RU", "stator_winding": "SW", "healthy": "HH"}
SWAP = {5: 7, 7: 5, 6: 8, 8: 6}


def read_tables(tables_dir):
    t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(tables_dir, "*_feature_table.csv")))],
                  ignore_index=True)
    h = pd.read_csv(os.path.join(tables_dir, "halves_features.csv"))
    keep = F6 + ["healthy"]
    return t[t.fault.isin(keep)].reset_index(drop=True), h[h.fault.isin(keep)].reset_index(drop=True)


def positive_columns(*tables):
    allv = pd.concat([x for x in tables], ignore_index=True)
    return {c for c in allv.columns if "__" in c and pd.api.types.is_numeric_dtype(allv[c])
            and (allv[c].dropna() > 0).all() and allv[c].notna().any()}


def add_derived(tab, names, keys):
    """adds the derived columns; healthy reference = healthy row with the same keys"""
    hh = {tuple(r[k] for k in keys): r.to_dict() for _, r in tab[tab.fault == "healthy"].iterrows()}
    rows = [M.derive(r.to_dict(), hh.get(tuple(r[k] for k in keys)), names) for _, r in tab.iterrows()]
    return pd.concat([tab.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def load(tables_dir, names=None, positive=None):
    t, h = read_tables(tables_dir)
    if names is None:
        positive = positive or positive_columns(t, h)
        names = M.derived_names(M.base_features(t.columns, positive))
    return add_derived(t, names, ["condition_id"]), add_derived(h, names, ["condition_id", "half"]), names


def hand_trigger_pairs():
    """(fault, raw feature) used by any hand-found trigger, in any group"""
    defs = (F.TRIGGER_DEFINITIONS + F.TRIGGER_DEFINITIONS_V3 + F.TRIGGERS_UNLOADED_ONLY + F.TRIGGERS_LOADED_ONLY)
    return {(d[1], d[2]) for d in defs}


def looks(full, halves, c):
    """independent check data for speed c: both 5 s halves, or the reverse ramp"""
    if c <= 4:
        return [halves[(halves.half == x) & (halves.condition_id == c)] for x in ("A", "B")]
    return [full[full.condition_id == SWAP[c]]]
