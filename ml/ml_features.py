"""
ml_features.py
===============
The feature space the ML search works in. Pure numpy: no scikit-learn, so
it can run wherever the trigger code runs.

Two kinds of derived features, both built from the normal feature vector
(fault_feature_extraction.build_feature_vector) and the healthy motor's
vector at the same speed and load:

  rel__<channel>__<feature>
      log( value / healthy value )
      "how many times the healthy level", on a log scale. Comparable across
      speeds, which lets one rule cover several speeds.

  xch__<chA>_<chB>__<feature>
      log( value_A / value_B ) - log( healthy_A / healthy_B )
      ratio of the same feature on two sensors, relative to the same ratio
      on the healthy motor. Describes WHERE / in which direction the motor
      shakes more, which no hand-found trigger looks at.

Only strictly positive features are used (logs). Temperature is left out
(the 7 C rule handles it), as are bookkeeping features (peak frequencies,
counts, dominant-band indices).
"""
import itertools

import numpy as np

CHANNELS = ["accel1", "accel2", "accel3", "mic"]
EXCLUDE = ("temp__", "fft_f0_hz", "shaft_hz", "fft_n_significant", "dwt_dominant", "fft_valley",
           "sk_peak_freq_hz", "psd_peak_hz", "concentration_index", "rise_vs_healthy")
EPS = 1e-12


def base_features(columns, positive):
    """base feature names (part after 'ch__') present on all four channels and positive"""
    per = {}
    for c in columns:
        if "__" not in c or any(e in c for e in EXCLUDE):
            continue
        ch, base = c.split("__", 1)
        if ch in CHANNELS and c in positive:
            per.setdefault(base, set()).add(ch)
    return sorted(b for b, chs in per.items() if len(chs) == len(CHANNELS))


def derived_names(bases):
    rel = [f"rel__{ch}__{b}" for b in bases for ch in CHANNELS]
    xch = [f"xch__{a}_{b2}__{b}" for b in bases for a, b2 in itertools.combinations(CHANNELS, 2)]
    return rel + xch


def _lg(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return np.log(v) if v > 0 else np.nan


def derive(vec, healthy, names):
    """vec, healthy: dicts of raw features. names: derived features wanted."""
    out = {}
    for n in names:
        kind, rest = n.split("__", 1)
        if kind == "rel":
            ch, b = rest.split("__", 1)
            k = f"{ch}__{b}"
            out[n] = _lg(vec.get(k)) - _lg(healthy.get(k)) if healthy else np.nan
        elif kind == "xch":
            pair, b = rest.split("__", 1)
            a, c = pair.split("_")
            ka, kc = f"{a}__{b}", f"{c}__{b}"
            out[n] = ((_lg(vec.get(ka)) - _lg(vec.get(kc))) - (_lg(healthy.get(ka)) - _lg(healthy.get(kc)))
                      if healthy else np.nan)
    return out


def raw_needed(names):
    """raw feature names a list of derived features depends on (for the healthy reference)"""
    need = set()
    for n in names:
        kind, rest = n.split("__", 1)
        if kind == "rel":
            need.add(rest)
        else:
            pair, b = rest.split("__", 1)
            for ch in pair.split("_"):
                need.add(f"{ch}__{b}")
    return sorted(need)


def describe(n):
    """plain-language name of a derived feature"""
    kind, rest = n.split("__", 1)
    if kind == "rel":
        ch, b = rest.split("__", 1)
        return f"{ch} {b} vs healthy"
    pair, b = rest.split("__", 1)
    a, c = pair.split("_")
    return f"{b}: {a} / {c} ratio vs healthy"
