"""Does each unloaded trigger still work under load?"""
import sys, warnings, pandas as pd, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _data import load_combined
warnings.filterwarnings("ignore")
import fault_feature_extraction as F
T = load_combined(sys.argv[1])
U, L = T[T.load == 0], T[T.load == 1]
calU, calL = F.calibrate_triggers(U), F.calibrate_triggers(L)
rows = []
for tid, rec in calU.items():
    for c, rU in rec["by_condition"].items():
        if not rU["usable"]:
            continue
        rL = calL[tid]["by_condition"].get(c)
        # transfer: apply unloaded rule to loaded runs
        sub = L[(L.condition_id == c) & (L.fault != "healthy")]
        hit_true = sub[sub.fault == rec["fault"]][rec["feature"]].apply(lambda x: F._check(rU, x)).sum()
        fa = sub[sub.fault != rec["fault"]][rec["feature"]].apply(lambda x: F._check(rU, x)).sum()
        same_shape = rL is not None and rL["op"] == rU["op"]
        rows.append(dict(trigger=tid, fault=rec["fault"], feature=rec["feature"], cond=c,
                         unl_value=rU["value"], load_value=rL["value"] if rL else np.nan,
                         unl_op=rU["op"], load_op=rL["op"] if rL else None,
                         load_sep=rL["separation"] if rL else np.nan,
                         load_usable=bool(rL and rL["usable"]), same_shape=same_shape,
                         transfer_hit=int(hit_true), transfer_false_alarms=int(fa)))
R = pd.DataFrame(rows)
R["pattern_holds"] = R.same_shape & R.load_usable
R["transfer_ok"] = (R.transfer_hit == 1) & (R.transfer_false_alarms == 0)
R.to_csv(sys.argv[2], index=False)
g = R.groupby(["trigger", "fault"]).agg(conds=("cond", lambda s: ",".join(map(str, s))),
        n=("cond", "size"), holds=("pattern_holds", "sum"), transfer=("transfer_ok", "sum")).reset_index()
pd.set_option("display.width", 200)
print(g.sort_values(["fault", "trigger"]).to_string(index=False))
print("\nrule-level: pattern holds %d/%d, unloaded threshold works as-is %d/%d" % (R.pattern_holds.sum(), len(R), R.transfer_ok.sum(), len(R)))
