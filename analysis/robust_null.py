"""How many cross-load 'replications' would we get by chance?
Shuffle which loaded recording belongs to which fault (within each speed),
redo the search, repeat. Compare with the real count, per fault.

    python3 analysis/robust_null.py <package folder> [min_gap 0.15] [n_shuffles 100]

<package folder> holds tables_unloaded/ and tables_loaded/ (see _data.py)."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _data import load_combined
warnings.filterwarnings("ignore")
T = load_combined(sys.argv[1])
EXCL = ("temp__", "stat_mean", "stat_median", "fft_f0_hz", "shaft_hz", "fft_n_significant", "dwt_dominant", "fft_valley")
num = [c for c in T.columns if T[c].dtype != object and "__" in c and not any(e in c for e in EXCL)]
FAULTS = ["BR", "FB", "KA", "RM", "RU", "SW", "VU"]
SEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
pos = {f: bool((T[f].dropna() > 0).all()) for f in num}

def ext(load_df):
    """dict (cond, feature, dir) -> (fault, sep)"""
    out = {}
    for c, g in load_df.groupby("condition_id"):
        g = g.set_index("c")[num]
        for f in num:
            v = g[f].dropna()
            if len(v) < 7 or v.nunique() < 7: continue
            tv = np.log(v) if pos[f] else v
            sp = tv.max() - tv.min()
            s = tv.sort_values()
            for d, fa, gap in (("high", s.index[-1], s.iloc[-1] - s.iloc[-2]), ("low", s.index[0], s.iloc[1] - s.iloc[0])):
                sep = gap if pos[f] else gap / sp
                if sep >= SEP: out[(c, f, d)] = fa
    return out

F7 = T[T.c.isin(FAULTS)]
EU = ext(F7[F7.load == 0])
L = F7[F7.load == 1].copy()

def count(EL):
    cnt = {f: 0 for f in FAULTS}
    for k, fa in EU.items():
        if EL.get(k) == fa: cnt[fa] += 1
    return cnt

real = count(ext(L))
rng = np.random.default_rng(0)
null = []
for i in range(int(sys.argv[3]) if len(sys.argv) > 3 else 100):
    Lp = L.copy()
    for c in range(1, 9):
        m = Lp.condition_id == c
        Lp.loc[m, "c"] = rng.permutation(Lp.loc[m, "c"].values)
    null.append(count(ext(Lp)))
N = pd.DataFrame(null)
res = pd.DataFrame({"real": pd.Series(real), "chance_mean": N.mean().round(1),
                    "chance_95pct": N.quantile(0.95), "p_value": [(N[f] >= real[f]).mean() for f in FAULTS]})
print(f"sep >= {SEP}; unloaded extremes: {len(EU)}")
print(res.to_string()); print("total real", sum(real.values()), "chance", N.sum(axis=1).mean().round(1), "95%", N.sum(axis=1).quantile(.95))
