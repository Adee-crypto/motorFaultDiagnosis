"""Shared loader for the analysis scripts.

load_combined(path) returns one table with both loads (128 runs), a 'load'
column (0/1) and a 'c' column with the two-letter fault code.
path is either
  - the package folder that contains tables_unloaded/ and tables_loaded/, or
  - a .pkl file with that combined table (as used during the analysis).
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:          # so 'import fault_feature_extraction' works
    sys.path.insert(0, ROOT)

CODE = {"healthy": "HH", "bowed_rotor": "BR", "faulty_bearing": "FB", "broken_rotor_bars": "KA",
        "rotor_misalignment": "RM", "rotor_unbalance": "RU", "stator_winding": "SW",
        "voltage_unbalance": "VU"}


def load_combined(path):
    if str(path).endswith(".pkl"):
        return pd.read_pickle(path).reset_index(drop=True)
    parts = []
    for sub, load in (("tables_unloaded", 0), ("tables_loaded", 1)):
        files = sorted(glob.glob(os.path.join(path, sub, "*_feature_table.csv")))
        if not files:
            sys.exit(f"no *_feature_table.csv in {os.path.join(path, sub)}")
        t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        t["load"] = load
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    T["c"] = T.fault.map(CODE)
    return T
