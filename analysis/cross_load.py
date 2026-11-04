"""Cross-load identification: fit on one load, test on the other.
E1  nearest fingerprint: each test run is matched to the fault whose
    run at the same speed (other load) is closest, over all features
    expressed relative to the healthy motor at that speed and load.
E2  deck triggers, thresholds fitted on the other load, using features
    relative to healthy (so a load-wide shift cancels out).
E3  deck triggers on raw features (the plain transfer test)."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _data import load_combined
warnings.filterwarnings("ignore")
import fault_feature_extraction as F
T = load_combined(sys.argv[1])
EXCL = ("temp__", "stat_mean", "stat_median", "fft_f0_hz", "shaft_hz", "fft_n_significant", "dwt_dominant", "fft_valley")
num = [c for c in T.columns if pd.api.types.is_numeric_dtype(T[c]) and "__" in c]
use = [c for c in num if not any(e in c for e in EXCL)]
FAULTS = ["BR", "FB", "KA", "RM", "RU", "SW", "VU"]
pos = {f: bool((T[f].dropna() > 0).all()) for f in num}
R = T.copy()
for (ld, c), g in T.groupby(["load", "condition_id"]):
    hh = g[g.c == "HH"].iloc[0]
    for f in num:
        R.loc[g.index, f] = np.log(g[f] / hh[f]) if pos[f] else g[f] - hh[f]

def e1(fit, test, faults):
    out = []
    for c in range(1, 9):
        A = fit[(fit.condition_id == c) & fit.c.isin(faults)].set_index("c")[use]
        B = test[(test.condition_id == c) & test.c.isin(faults)].set_index("c")[use]
        cols = [x for x in use if A[x].notna().all() and B[x].notna().all()]
        sd = pd.concat([A[cols], B[cols]]).std().replace(0, np.nan)
        cols = [x for x in cols if pd.notna(sd[x])]
        za, zb = A[cols] / sd[cols], B[cols] / sd[cols]
        for f in zb.index:
            d = ((za - zb.loc[f]) ** 2).sum(axis=1)
            out.append(dict(c=f, cond=c, pred=d.idxmin(), dist=d.to_dict()))
    return pd.DataFrame(out)

def e23(fit, test, faults, relative):
    lab = {k: v for k, v in zip(["bowed_rotor","faulty_bearing","broken_rotor_bars","rotor_misalignment","rotor_unbalance","stator_winding","voltage_unbalance"], FAULTS)}
    defs = [d for d in F.TRIGGER_DEFINITIONS if lab[d[1]] in faults and not d[2].startswith("temp__")] if relative else \
           [d for d in F.TRIGGER_DEFINITIONS if lab[d[1]] in faults]
    fitd = fit[fit.c.isin(faults)]
    cal = F.calibrate_triggers(fitd, definitions=defs)
    out = []
    for _, r in test[test.c.isin(faults)].iterrows():
        s, _, b = F.score_triggers(r.to_dict(), int(r.condition_id), cal)
        out.append(dict(c=r.c, cond=int(r.condition_id), pred=lab.get(b, "none"),
                        pts={lab[k]: v["points"] for k, v in s.items()}))
    return pd.DataFrame(out)

def combined(res, kind):
    ok = 0; lines = []
    for f, g in res.groupby("c"):
        tot = {}
        for _, r in g.iterrows():
            if kind == "e1":   # vote: closest = 1 point
                tot[r.pred] = tot.get(r.pred, 0) + 1
            else:
                for k, v in r.pts.items(): tot[k] = tot.get(k, 0) + v
        best = max(tot, key=tot.get) if tot and max(tot.values()) > 0 else "none"
        ok += best == f; lines.append(f"{f}->{best}")
    return ok, " ".join(lines)

for faults, tag in ((FAULTS, "7 faults"), ([f for f in FAULTS if f != "VU"], "6 faults (VU dropped)")):
    print(f"\n######## {tag}")
    for fl, tl, name in ((0, 1, "fit unloaded -> test loaded"), (1, 0, "fit loaded -> test unloaded")):
        for kind, fn in (("e1", lambda a, b: e1(R[R.load == a], R[R.load == b], faults)),
                         ("e2", lambda a, b: e23(R[R.load == a], R[R.load == b], faults, True)),
                         ("e3", lambda a, b: e23(T[T.load == a], T[T.load == b], faults, False))):
            res = fn(fl, tl)
            acc = (res.c == res.pred).mean()
            ok, lines = combined(res, kind)
            print(f"{name:28s} {kind}: per-run {acc:.0%}   8-speed combined {ok}/{len(faults)}  [{lines}]")
