"""
prune_ml.py
============
Removes repetition from the ML trigger list of ONE load, for every fault.

    python3 prune_ml.py <tables_dir> <load 0|1> <discover_out_dir> <pruned_out_dir>

Why: the trigger score is a vote, every fired trigger adds points to its
fault. Several ML triggers can be the same observation measured in slightly
different ways (e.g. "accelerometer 1 has more 5-10 kHz energy than
accelerometer 2" and "... more energy above 10 kHz than accelerometer 2").
Counting them all would give one piece of evidence many votes.

Rule, per fault:
  1. Order the fault's ML triggers by strength: number of speeds where the
     trigger passed the consistency check, then the forest's ranking.
  2. Two triggers REPEAT each other if
       - their features move together (|Spearman| > CORR over the 48 fault
         runs of this load), or
       - they share a sensor (e.g. "accelerometer 1" and "accelerometer 1 /
         accelerometer 2") and measure the same thing: overlapping frequency
         bands (e.g. the 1.3-2.6 kHz wavelet band and the 1-2 kHz PSD band),
         or the same kind of measure (spectral shape, impulsiveness, ...).
         Neighbouring, non-overlapping bands count as different evidence.
  3. Going down the list, a trigger keeps only the speeds that no stronger
     trigger it repeats already covers. With no speeds left it is dropped.
     A trigger that is not repeated by any stronger one is kept as is.
So unique evidence is always kept; repeated evidence is kept once per speed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ml_data as D

CORR = 0.7
DWT_BANDS = {"D4": (1300, 2600), "D3": (2600, 5200), "D2": (5200, 10400), "D1": (10400, 20833),
             "d4": (1300, 2600), "hf": (2600, 20833), "low": (0, 1300), "A4": (0, 1300)}


def measure(feature):
    """(sensor or sensor pair, what is measured) of an ML feature"""
    kind, rest = feature.split("__", 1)
    where, base = rest.split("__", 1)
    if base.startswith("psd_frac_"):
        a, b = base[len("psd_frac_"):].split("_")
        return where, ("band", float(a), float(b))
    if base.startswith("dwt_"):
        for k, band in DWT_BANDS.items():
            if f"_{k}_" in base or base.endswith(f"_{k}_frac") or f"dwt_{k}" in base:
                return where, ("band",) + band
    if "centroid" in base or "flatness" in base:
        return where, ("shape",)
    if base.startswith("sk_"):
        return where, ("impulsiveness",)
    return where, ("other", base)


def same_measure(f1, f2):
    w1, m1 = measure(f1)
    w2, m2 = measure(f2)
    if not set(w1.split("_")) & set(w2.split("_")):   # no sensor in common
        return False
    if m1[0] == "band" and m2[0] == "band":
        return m1[1] < m2[2] and m2[1] < m1[2]
    return m1 == m2


def main():
    tables, load, src, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    payload = json.load(open(os.path.join(src, f"ml_triggers_load{load}.json")))
    trigs = payload["triggers"]
    names = sorted({t["feature"] for t in trigs.values()})
    full, _, _ = D.load(tables, names)
    f6 = full[full.fault.isin(D.F6)]
    cand = pd.read_csv(os.path.join(src, f"ml_candidates_load{load}.csv"))
    rank = {(r.fault, r.feature): r["rank"] for _, r in cand.iterrows()}

    def corr(a, b):
        x, y = f6[a].values, f6[b].values
        ok = ~(np.isnan(x) | np.isnan(y))
        return abs(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 10 else 0.0

    kept, log = {}, []
    for fault in D.F6:
        mine = [(tid, t) for tid, t in trigs.items() if t["fault"] == fault]
        mine.sort(key=lambda kv: (-len(kv[1]["by_condition"]), rank.get((fault, kv[1]["feature"]), 999)))
        chosen = []
        for tid, t in mine:
            speeds = set(int(c) for c in t["by_condition"])
            reps = [(k, kt) for k, kt in chosen
                    if same_measure(t["feature"], kt["feature"]) or corr(t["feature"], kt["feature"]) > CORR]
            covered = set().union(*[set(int(c) for c in kt["by_condition"]) for _, kt in reps]) if reps else set()
            left = sorted(speeds - covered)
            if not reps:
                status = "kept (unique)"
            elif not left:
                status = "dropped: repeats " + ", ".join(k for k, _ in reps)
            else:
                status = f"trimmed to speeds {left}: other speeds repeat " + ", ".join(k for k, _ in reps)
            log.append({"load": load, "fault": fault, "trigger": tid, "feature": t["feature"], "meaning": t["note"],
                        "relation": t["relation"], "speeds": sorted(speeds), "status": status})
            if left:
                t2 = dict(t)
                t2["by_condition"] = {str(c): t["by_condition"][str(c)] for c in left}
                if t2["kind"] == "P" and len(left) < 3:
                    t2["kind"] = "S"
                chosen.append((tid, t2))
        kept.update(dict(chosen))

    # healthy reference: the ML features' raw inputs + Stage A channels
    need = sorted(set(D.M.raw_needed([t["feature"] for t in kept.values()])) |
                  {"accel1__stat_variance", "mic__stat_variance"})
    hh = full[full.fault == "healthy"]
    href = {int(r.condition_id): {k: float(r[k]) for k in need} for _, r in hh.iterrows()}
    with open(os.path.join(out, f"ml_triggers_load{load}.json"), "w") as fh:
        json.dump({"load": load, "pruned": True, "corr_limit": CORR, "triggers": kept, "healthy_reference": href}, fh, indent=1)
    L = pd.DataFrame(log)
    L.to_csv(os.path.join(out, f"ml_pruning_load{load}.csv"), index=False)
    s = L.assign(k=L.status.str.split(" ").str[0]).groupby(["fault", "k"]).size().unstack(fill_value=0)
    s["after"] = L.groupby("fault").apply(lambda g: sum(not x.startswith("dropped") for x in g.status))
    print(f"load {load}: {len(trigs)} ML triggers -> {len(kept)} after pruning")
    print(s.to_string())


if __name__ == "__main__":
    main()
