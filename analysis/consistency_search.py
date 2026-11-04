"""
consistency_search.py
======================
Looks for fault-unique patterns in ONE load condition and keeps only the
consistent ones.

    python3 consistency_search.py <full_tables_dir> <halves_csv> <other_load_tables_dir> <out_csv> [n_shuffles]

1. Discover  -- on the full 10 s runs, at each speed: is one fault the
   highest (or lowest) of the six faults on a feature, with a clear gap
   (>= 35% on a log scale for positive features, >= 15% of the spread for
   signed ones)?
2. Consistent -- the same fault, same direction, still extreme with a gap
   of >= 10% in an independent look at the same speed:
     constant speeds (1-4): BOTH 5 s halves of the run
     ramps (5-8): the reverse-direction ramp, a separate recording
3. Other load -- does the same pattern also hold in the other load
   (>= 10% gap)? -> "both loads" vs "this load only".
4. Chance level -- fault labels in the check data are shuffled
   n_shuffles times; how many discoveries pass step 2 by luck?
Temperature is left out here (handled by the >7 C rule).
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
AB = {"bowed_rotor": "BR", "faulty_bearing": "FB", "broken_rotor_bars": "KA",
      "rotor_misalignment": "RM", "rotor_unbalance": "RU", "stator_winding": "SW"}
F6 = list(AB.values())
SWAP = {5: 7, 7: 5, 6: 8, 8: 6}
EXCL = ("temp__", "fft_f0_hz", "shaft_hz", "fft_n_significant", "dwt_dominant", "fft_valley",
        "sk_peak_freq_hz", "psd_peak_hz", "concentration_index")
D_POS, D_SIGNED, C_POS, C_SIGNED = 0.30, 0.15, 0.10, 0.05


def load_tables(d):
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*_feature_table.csv"))], ignore_index=True)
    t["c"] = t.fault.map(AB)
    return t[t.c.isin(F6)]


def extremes(g, feats, pos):
    """g: one speed, index = fault code. -> {(feature, dir): (fault, sep)}"""
    out = {}
    for f in feats:
        v = g[f].dropna()
        if len(v) < len(F6) or v.nunique() < len(F6):
            continue
        tv = np.log(v) if pos[f] else v
        sp = tv.max() - tv.min()
        if sp <= 0:
            continue
        s = tv.sort_values()
        for d, fa, gap in (("high", s.index[-1], s.iloc[-1] - s.iloc[-2]), ("low", s.index[0], s.iloc[1] - s.iloc[0])):
            out[(f, d)] = (fa, gap if pos[f] else gap / sp)
    return out


def main():
    full, halves, other, out_csv = sys.argv[1:5]
    nsh = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    T = load_tables(full)
    H = pd.read_csv(halves); H["c"] = H.fault.map(AB); H = H[H.c.isin(F6)]
    O = load_tables(other)
    feats = [c for c in T.columns if "__" in c and pd.api.types.is_numeric_dtype(T[c]) and not any(e in c for e in EXCL)]
    allv = pd.concat([T[feats], O[feats], H[feats]])
    pos = {f: bool((allv[f].dropna() > 0).all()) for f in feats}

    def views(tab, c):
        return tab[tab.condition_id == c].set_index("c")

    disc = []
    for c in range(1, 9):
        for (f, d), (fa, sep) in extremes(views(T, c), feats, pos).items():
            if sep >= (D_POS if pos[f] else D_SIGNED):
                disc.append((c, f, d, fa, sep))

    def check_sets(labels_perm=None):
        """the independent looks per speed, optionally with shuffled labels"""
        looks = {}
        for c in range(1, 9):
            if c <= 4:
                parts = [views(H[H.half == h], c) for h in ("A", "B")]
            else:
                parts = [views(T, SWAP[c])]
            if labels_perm is not None:
                parts = [p.set_axis(labels_perm(c, i, list(p.index)), axis=0) for i, p in enumerate(parts)]
            looks[c] = [extremes(p, feats, pos) for p in parts]
        return looks

    def passes(looks, c, f, d, fa):
        need = C_POS if pos[f] else C_SIGNED
        return all(L.get((f, d), (None, 0))[0] == fa and L[(f, d)][1] >= need for L in looks[c])

    looks = check_sets()
    oth = {c: extremes(views(O, c), feats, pos) for c in range(1, 9)}
    rows = []
    for c, f, d, fa, sep in disc:
        ok = passes(looks, c, f, d, fa)
        o = oth[c].get((f, d), (None, 0))
        both = o[0] == fa and o[1] >= (C_POS if pos[f] else C_SIGNED)
        rows.append(dict(cond=c, feature=f, dir=d, fault=fa, sep=sep, consistent=ok, other_load=both,
                         check_min_sep=min(L.get((f, d), (None, 0))[1] if L.get((f, d), (None, 0))[0] == fa else 0 for L in looks[c])))
    R = pd.DataFrame(rows)
    R["group"] = np.where(~R.consistent, "inconsistent", np.where(R.other_load, "both loads", "this load only"))
    R.to_csv(out_csv, index=False)

    real = R[R.consistent].groupby("fault").size().reindex(F6, fill_value=0)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(nsh):
        perms = {}

        def perm(c, i, idx):
            return list(rng.permutation(idx))
        lk = check_sets(perm)
        cnt = {f: 0 for f in F6}
        for c, f, d, fa, sep in disc:
            if passes(lk, c, f, d, fa):
                cnt[fa] += 1
        null.append(cnt)
    N = pd.DataFrame(null)
    res = pd.DataFrame({"discovered": R.groupby("fault").size().reindex(F6, fill_value=0),
                        "consistent": real, "chance_mean": N.mean().round(1), "chance_95": N.quantile(.95),
                        "p": [(N[f] >= real[f]).mean() for f in F6],
                        "this_load_only": R[R.group == "this load only"].groupby("fault").size().reindex(F6, fill_value=0),
                        "both_loads": R[R.group == "both loads"].groupby("fault").size().reindex(F6, fill_value=0)})
    print(res.to_string())
    pd.set_option("display.width", 220)
    for fa in F6:
        top = R[(R.fault == fa) & R.consistent].sort_values("check_min_sep", ascending=False).head(8)
        print(f"-- {fa}")
        for _, r in top.iterrows():
            print(f"   cond {r.cond} {r['dir']:4s} {r.feature:38s} sep={r.sep:.2f} check={r.check_min_sep:.2f} {r.group}")


if __name__ == "__main__":
    main()
