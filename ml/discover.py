"""
discover.py
============
ML pattern search for ONE load, kept separate from the hand-found triggers.

    python3 discover.py <tables_dir> <other_load_tables_dir> <load 0|1> <out_dir> [n_shuffles]

<tables_dir> has the *_feature_table.csv files and halves_features.csv of
the load being searched (process_fault.py / validate_triggers.py --halves).

Steps
 1. Feature space: every vibration/sound feature relative to the healthy
    motor at the same speed and load (rel__), and every same-feature ratio
    between two sensors (xch__). See ml_features.py. Temperature is left out.
 2. Model: random forest, trained with LEAVE-ONE-SPEED-OUT: it must
    recognise each fault at a speed it never saw. Accuracy is compared with
    the same model trained on shuffled labels (chance level).
 3. Ranking: per fault, a one-vs-rest forest gives each feature an
    importance (averaged over the 8 leave-one-speed-out fits).
 4. Novelty filter: a feature is dropped if a hand-found trigger already
    uses it for the same fault, or if it moves together with one
    (|Spearman| > 0.9 over the fault runs). Near-duplicates among the ML
    candidates themselves are also dropped.
 5. Rules, in two forms the hand search never used:
      P  one threshold that holds at 3 or more speeds (speed-independent)
      S  per-speed threshold, only for sensor ratios (xch__)
    A rule needs the fault to be the clear extreme (gap >= 0.30 in log
    units, ~35%, the same bar as the hand search).
 6. Verification, same tests as the hand-found triggers:
      consistency : the fitted rule fires for its fault and for no other
                    fault in both 5 s halves (constant speeds) or in the
                    reverse ramp (ramps)
      chance      : fault labels of the check data shuffled n times
      other load  : does the rule (same thresholds) or the pattern
                    (re-fitted) also hold in the other load?
 Outputs in <out_dir>: ml_accuracy_load<L>.txt, ml_candidates_load<L>.csv,
 ml_rules_load<L>.csv, ml_chance_load<L>.csv, ml_triggers_load<L>.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

import ml_data as D
import ml_features as M

warnings.filterwarnings("ignore")
GAP_DISCOVER, GAP_CHECK, MIN_POOLED = 0.30, 0.10, 3
TOP_K, CORR_MAX = 20, 0.9
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- model
def _xy(full, names, labels=None):
    f = full[full.fault.isin(D.F6)].reset_index(drop=True)
    X = f[names].astype(float)
    X = X.fillna(X.median()).fillna(0.0).values
    y = np.array(labels if labels is not None else f.fault.values)
    return X, y, f.condition_id.values


def loso_accuracy(X, y, cond, seed=0, n_trees=300):
    hits = 0
    for c in np.unique(cond):
        tr, te = cond != c, cond == c
        m = RandomForestClassifier(n_trees, random_state=seed, class_weight="balanced", n_jobs=-1).fit(X[tr], y[tr])
        hits += (m.predict(X[te]) == y[te]).sum()
    return hits / len(y)


def shuffled_within_speed(y, cond):
    y2 = y.copy()
    for c in np.unique(cond):
        i = np.where(cond == c)[0]
        y2[i] = RNG.permutation(y2[i])
    return y2


def importance(X, y, cond, names, fault, n_trees=300):
    imp = np.zeros(len(names))
    for c in np.unique(cond):
        tr = cond != c
        m = RandomForestClassifier(n_trees, random_state=0, class_weight="balanced", n_jobs=-1)
        m.fit(X[tr], (y[tr] == fault).astype(int))
        imp += m.feature_importances_
    return imp / len(np.unique(cond))


# ---------------------------------------------------------------- novelty
def rel_value(tab, feat):
    """a hand-trigger feature relative to healthy at the same speed (log ratio or difference)"""
    hh = tab[tab.fault == "healthy"].set_index("condition_id")[feat]
    base = tab.condition_id.map(hh)
    v = tab[feat].astype(float)
    if (pd.concat([v, base]).dropna() > 0).all():
        return np.log(v) - np.log(base)
    return v - base


def novelty_filter(full, ranked, fault, hand_pairs):
    f_rows = full.fault.isin(D.F6)
    hand_feats = [ft for (fa, ft) in hand_pairs if fa == fault and ft in full.columns]
    hand_vals = {ft: rel_value(full, ft)[f_rows].values for ft in hand_feats}
    kept, dropped = [], []
    for n in ranked:
        if n.startswith("rel__") and (fault, n.split("__", 1)[1]) in hand_pairs:
            dropped.append((n, "same feature as a hand trigger"))
            continue
        x = full.loc[f_rows, n].values
        why = None
        for ft, hv in hand_vals.items():
            ok = ~(np.isnan(x) | np.isnan(hv))
            if ok.sum() > 10 and abs(spearmanr(x[ok], hv[ok])[0]) > CORR_MAX:
                why = f"moves with hand trigger feature {ft}"
                break
        if why is None:
            for k in kept:
                y = full.loc[f_rows, k].values
                ok = ~(np.isnan(x) | np.isnan(y))
                if ok.sum() > 10 and abs(spearmanr(x[ok], y[ok])[0]) > CORR_MAX:
                    why = f"near-duplicate of ML candidate {k}"
                    break
        (dropped.append((n, why)) if why else kept.append(n))
        if len(kept) >= TOP_K:
            break
    return kept, dropped


# ---------------------------------------------------------------- rules
def fault_matrix(full, feat):
    f = full[full.fault.isin(D.F6)]
    return f.pivot_table(index="fault", columns="condition_id", values=feat)


def extreme_at(col, fault, gap):
    """('high'|'low', threshold) if fault is the clear extreme in this column"""
    col = col.dropna()
    if fault not in col.index or len(col) < len(D.F6):
        return None
    v, o = col[fault], col.drop(fault)
    if v - o.max() >= gap:
        return "high", (v + o.max()) / 2
    if o.min() - v >= gap:
        return "low", (v + o.min()) / 2
    return None


def pooled_rule(mat, fault):
    """one threshold valid at >= MIN_POOLED speeds; returns (dir, thr, speeds) or None"""
    best = None
    for d in ("high", "low"):
        iv = []
        for c in mat.columns:
            col = mat[c].dropna()
            if fault not in col.index or len(col) < len(D.F6):
                continue
            v, o = col[fault], col.drop(fault)
            lo, hi = ((o.max() + GAP_DISCOVER / 2, v - GAP_DISCOVER / 2) if d == "high"
                      else (v + GAP_DISCOVER / 2, o.min() - GAP_DISCOVER / 2))
            if lo <= hi:
                iv.append((lo, hi, int(c)))
        pts = sorted({p for a, b, _ in iv for p in (a, b)})
        for p in pts:
            S = [c for a, b, c in iv if a <= p <= b]
            if len(S) >= MIN_POOLED and (best is None or len(S) > len(best[2])):
                los = [a for a, b, c in iv if c in S]
                his = [b for a, b, c in iv if c in S]
                best = (d, (max(los) + min(his)) / 2, sorted(S))
    return best


def make_rule(d, thr):
    return {"op": ">=", "lo": float(thr), "hi": None} if d == "high" else {"op": "<=", "lo": None, "hi": float(thr)}


def fires(rule, x):
    if x is None or np.isnan(x):
        return False
    return x >= rule["lo"] if rule["op"] == ">=" else x <= rule["hi"]


def build_rules(full, cands):
    """cands: {fault: [features]} -> list of rule dicts"""
    rules = []
    for fault, feats in cands.items():
        for feat in feats:
            mat = fault_matrix(full, feat)
            p = pooled_rule(mat, fault)
            used = set()
            if p:
                d, thr, S = p
                rules.append({"fault": fault, "feature": feat, "kind": "P", "dir": d,
                              "by_condition": {c: make_rule(d, thr) for c in S}})
                used = set(S)
            if feat.startswith("xch__"):
                by = {}
                for c in mat.columns:
                    if int(c) in used:
                        continue
                    e = extreme_at(mat[c], fault, GAP_DISCOVER)
                    if e:
                        by[int(c)] = make_rule(*e)
                if by:
                    rules.append({"fault": fault, "feature": feat, "kind": "S",
                                  "dir": next(iter(by.values()))["op"], "by_condition": by})
    return rules


# ---------------------------------------------------------------- checks
def consistent_speeds(rule, full, halves, perm=None):
    ok = []
    for c, r in rule["by_condition"].items():
        good = True
        for i, L in enumerate(D.looks(full, halves, c)):
            L = L[L.fault.isin(D.F6)]
            labels = L.fault.values if perm is None else perm(c, i, L.fault.values)
            hit = {lab: fires(r, x) for lab, x in zip(labels, L[rule["feature"]].values)}
            if not hit.get(rule["fault"], False) or any(v for k, v in hit.items() if k != rule["fault"]):
                good = False
                break
        if good:
            ok.append(c)
    return ok


def relation(rule, hand_pairs):
    """'new idea' unless the ML feature is built from a feature a hand trigger
    already uses for the same fault (then it extends that trigger)"""
    used = {ft for fa, ft in hand_pairs if fa == rule["fault"]}
    parts = M.raw_needed([rule["feature"]])
    hit = [p for p in parts if p in used]
    return ("extends hand trigger feature " + ", ".join(hit)) if hit else "new idea"


def kept(rule, ok):
    return len(ok) >= (MIN_POOLED if rule["kind"] == "P" else 1)


def other_load(rule, ok, ofull):
    same, pattern = [], []
    for c in ok:
        col = ofull[ofull.condition_id == c].set_index("fault")[rule["feature"]]
        col = col[col.index.isin(D.F6)]
        r = rule["by_condition"][c]
        h = {k: fires(r, x) for k, x in col.items()}
        if h.get(rule["fault"]) and not any(v for k, v in h.items() if k != rule["fault"]):
            same.append(c)
        e = extreme_at(col, rule["fault"], GAP_CHECK)
        if e and (e[0] == "high") == (r["op"] == ">="):
            pattern.append(c)
    return same, pattern


# ---------------------------------------------------------------- main
def main():
    tables, other, load, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    nsh = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    os.makedirs(out, exist_ok=True)
    full, halves, names = D.load(tables)
    ofull, _, _ = D.load(other, names)
    hand = D.hand_trigger_pairs()
    tag = f"load{load}"
    log = []

    # 2. leave-one-speed-out accuracy vs chance
    X, y, cond = _xy(full, names)
    acc = loso_accuracy(X, y, cond)
    null = [loso_accuracy(X, shuffled_within_speed(y, cond), cond, seed=i, n_trees=150) for i in range(30)]
    log.append(f"Random forest, leave-one-speed-out, {len(names)} ML features, {len(y)} fault runs")
    log.append(f"  accuracy on the unseen speed: {acc:.0%}   (chance: mean {np.mean(null):.0%}, "
               f"95th pct {np.quantile(null, .95):.0%}, p = {np.mean(np.array(null) >= acc):.2f})")

    # 3-4. ranking + novelty
    cand, crow = {}, []
    for fault in D.F6:
        imp = importance(X, y, cond, names, fault)
        order = [names[i] for i in np.argsort(-imp)]
        keep, drop = novelty_filter(full, order, fault, hand)
        cand[fault] = keep
        rank = {n: i + 1 for i, n in enumerate(order)}
        for n in keep:
            crow.append({"fault": fault, "feature": n, "rank": rank[n], "importance": imp[names.index(n)],
                         "status": "candidate", "reason": "", "meaning": M.describe(n)})
        for n, why in drop:
            crow.append({"fault": fault, "feature": n, "rank": rank[n], "importance": imp[names.index(n)],
                         "status": "dropped", "reason": why, "meaning": M.describe(n)})
    pd.DataFrame(crow).sort_values(["fault", "rank"]).to_csv(os.path.join(out, f"ml_candidates_{tag}.csv"), index=False)

    # 5-6. rules and checks
    rules = build_rules(full, cand)
    rows, final = [], []
    for r in rules:
        ok = consistent_speeds(r, full, halves)
        same, pat = other_load(r, ok, ofull)
        r["consistent"], r["kept"] = ok, kept(r, ok)
        r["relation"] = relation(r, hand)
        rows.append({"fault": r["fault"], "feature": r["feature"], "kind": r["kind"], "dir": r["dir"],
                     "speeds_found": sorted(r["by_condition"]), "speeds_consistent": ok, "kept": r["kept"],
                     "other_load_same_threshold": same, "other_load_pattern": pat, "relation": r["relation"],
                     "meaning": M.describe(r["feature"])})
        if r["kept"]:
            final.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, f"ml_rules_{tag}.csv"), index=False)

    # chance: shuffle the fault labels of the check data
    def count(perm=None):
        cnt = {f: 0 for f in D.F6}
        for r in rules:
            if kept(r, consistent_speeds(r, full, halves, perm)):
                cnt[r["fault"]] += 1
        return cnt
    real = count()
    nulls = []
    for _ in range(nsh):
        def perm(c, i, labels):
            return RNG.permutation(labels)
        nulls.append(count(perm))
    N = pd.DataFrame(nulls)
    C = pd.DataFrame({"rules_found": R.groupby("fault").size().reindex(D.F6, fill_value=0),
                      "kept_real": pd.Series(real), "chance_mean": N.mean().round(2),
                      "chance_95pct": N.quantile(.95), "p": [(N[f] >= real[f]).mean() for f in D.F6]})
    tot = sum(real.values())
    C.loc["ALL"] = [len(R), tot, round(N.sum(axis=1).mean(), 2), N.sum(axis=1).quantile(.95),
                    (N.sum(axis=1) >= tot).mean()]
    C.to_csv(os.path.join(out, f"ml_chance_{tag}.csv"))
    log.append("\nRules kept after the consistency check, vs shuffled check labels:")
    log.append(C.to_string())

    # trigger file (same layout the trigger engine reads)
    trig = {}
    for i, r in enumerate(final):
        tid = f"ML{load}_{D.AB[r['fault']]}_{r['kind']}{i:02d}"
        trig[tid] = {"fault": r["fault"], "feature": r["feature"], "weight": 0.5, "group": "ml", "kind": r["kind"],
                     "note": M.describe(r["feature"]), "relation": r["relation"],
                     "by_condition": {int(c): dict(r["by_condition"][c], usable=True) for c in r["consistent"]}}
    hh = full[full.fault == "healthy"]
    need = M.raw_needed([t["feature"] for t in trig.values()])
    href = {int(rw.condition_id): {k: float(rw[k]) for k in need} for _, rw in hh.iterrows()}
    with open(os.path.join(out, f"ml_triggers_{tag}.json"), "w") as fh:
        json.dump({"load": load, "triggers": trig, "healthy_reference": href}, fh, indent=1)
    log.append(f"\n{len(final)} ML triggers kept -> ml_triggers_{tag}.json")
    for tid, t in trig.items():
        log.append(f"  {tid:16s} {t['fault']:20s} {t['note']:52s} speeds {str(sorted(t['by_condition'])):18s} "
                   f"{'NEW' if t['relation'] == 'new idea' else 'extends hand'}")
    txt = "\n".join(log)
    open(os.path.join(out, f"ml_accuracy_{tag}.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
