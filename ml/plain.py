"""plain-language names for ML trigger features (used by the reports and the paper)"""
CH = {"accel1": "accel 1", "accel2": "accel 2", "accel3": "accel 3", "mic": "microphone"}
DWT = {"D4": "1.3–2.6 kHz", "d4": "1.3–2.6 kHz", "D3": "2.6–5.2 kHz", "D2": "5.2–10.4 kHz", "D1": "10.4–20.8 kHz",
       "hf": "above 2.6 kHz"}


def what(base):
    if base.startswith("psd_frac_"):
        a, b = base[9:].split("_")
        return f"power share {float(a)/1000:g}–{float(b)/1000:g} kHz"
    for k, v in DWT.items():
        if base in (f"dwt_{k}_energy_frac", f"dwt_{k}_frac"):
            return f"energy share {v}"
    return {"fft_spectral_flatness": "spectral flatness", "fft_spectral_centroid_hz": "spectral centre frequency",
            "psd_centroid_hz": "spectral centre frequency", "sk_rise_ratio": "impulsiveness rise with frequency (SK)",
            "sk_mean_above_transition": "high-frequency impulsiveness (SK)", "sk_peak_value": "peak impulsiveness (SK)",
            "sk_med_1_5k": "impulsiveness 1–5 kHz (SK)", "stat_crest_factor": "peakiness (crest factor)",
            "stat_variance": "variance"}.get(base, base)


def plain(feature, op):
    kind, rest = feature.split("__", 1)
    where, base = rest.split("__", 1)
    d = "higher" if op == ">=" else "lower"
    if kind == "rel":
        return f"{CH[where]} {what(base)}, {d} than other faults (vs healthy)"
    a, b = where.split("_")
    return f"{what(base)}: {CH[a]} ÷ {CH[b]}, {d} than other faults"
