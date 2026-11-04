"""Search for triggers that hold in BOTH load conditions.
For every fault, speed condition and feature: is the fault the highest
(or lowest) of the 7 faults in the unloaded set AND in the loaded set,
with a clear gap both times? Done on raw features and on features
relative to the healthy motor at the same speed and load."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _data import load_combined
warnings.filterwarnings("ignore")
T = load_combined(sys.argv[1])
EXCL = ("temp__", "stat_mean", "stat_median", "fft_f0_hz", "shaft_hz", "fft_n_significant", "dwt_dominant",
        "fft_valley")
num = [c for c in T.columns if T[c].dtype != object and "__" in c and not any(e in c for e in EXCL)]
FAULTS = ["BR", "FB", "KA", "RM", "RU", "SW", "VU"]

def rel_table(T):
    out = T.copy()
    for (ld, c), g in T.groupby(["load", "condition_id"]):
        hh = g[g.c == "HH"][num].iloc[0]
        idx = g.index
        for f in num:
            v = g[f]
            if (T[f].dropna() > 0).all():
                out.loc[idx, f] = np.log(v / hh[f])
            else:
                out.loc[idx, f] = v - hh[f]
    return out

def extremes(T, positive):
    rows = []
    for (ld, c), g in T[T.c.isin(FAULTS)].groupby(["load", "condition_id"]):
        g = g.set_index("c")
        for f in num:
            v = g[f].dropna()
            if len(v) < 7 or v.nunique() < 7:
                continue
            lg = positive.get(f, False)
            tv = np.log(v) if lg else v
            spread = tv.max() - tv.min()
            if spread <= 0:
                continue
            s = tv.sort_values()
            for fault, direction, gap in ((s.index[-1], "high", s.iloc[-1] - s.iloc[-2]),
                                         (s.index[0], "low", s.iloc[1] - s.iloc[0])):
                sep = gap if lg else gap / spread
                rows.append(dict(load=ld, cond=c, feature=f, fault=fault, dir=direction, sep=sep,
                                 value=v[fault]))
    return pd.DataFrame(rows)

for version in ("raw", "relative"):
    TT = T if version == "raw" else rel_table(T)
    positive = {f: bool((TT[f].dropna() > 0).all()) for f in num} if version == "raw" else {f: False for f in num}
    # relative features are log-ratios (already log) or differences -> judge gap vs spread
    E = extremes(TT, positive)
    thr = 0.10 if version == "raw" else 0.10
    E = E[E.sep >= (0.15 if version == "raw" else 0.15)]
    m = E[E.load == 0].merge(E[E.load == 1], on=["cond", "feature", "fault", "dir"], suffixes=("_U", "_L"))
    m["min_sep"] = m[["sep_U", "sep_L"]].min(axis=1)
    m = m.sort_values("min_sep", ascending=False)
    m.to_csv(f"robust_{version}.csv", index=False)
    print(f"\n===== {version}: {len(m)} trigger candidates hold in both loads =====")
    cov = m.groupby(["fault", "cond"]).size().unstack(fill_value=0).reindex(index=FAULTS, columns=range(1, 9), fill_value=0)
    print(cov.to_string())
    for fault in FAULTS:
        top = m[m.fault == fault].head(6)
        print(f"-- {fault}")
        for _, r in top.iterrows():
            print(f"   cond {r.cond}  {r.dir:4s} {r.feature:36s} sepU={r.sep_U:.2f} sepL={r.sep_L:.2f}")
