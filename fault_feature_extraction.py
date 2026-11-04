"""
fault_feature_extraction.py
============================
Turns the qualitative diagnostic patterns described in Fault_Analysis.pdf
into a numeric feature vector per channel, suitable for feeding a
classifier (a simple rule-based "distance to profile" scorer is included
at the bottom as a cold-start baseline).

Patterns being quantified (from the slides):
  Healthy         - clean FFT harmonics of f0, minimal energy elsewhere;
                     accel-3 DWT: D1+D2 bands > 60% of total energy
  Bowed rotor     - FFT dominated by a broad low-frequency peak, harmonics
                     suppressed; accel-1 DWT: A4 band > 96% of energy;
                     mic spectral kurtosis flat below ~10 kHz then rises
                     sharply to ~350-400 above it; temperature: highest
                     kurtosis of all conditions, peak > 35 C
  Faulty bearing  - accel-1 FFT: two peaks separated by a valley near
                     0 Hz and 2500 Hz; accel-1 variance orders of
                     magnitude above healthy (0-25); temperature: lowest
                     skewness of all conditions
  Broken rotor bars - accel-1 variance orders of magnitude above healthy,
                     roughly 2x the bearing case at max constant speed;
                     temperature rises >18 C, peak close to 50 C, under
                     accelerated/decelerated 15-45 Hz operation

Added in v2 (from the extended deck, slides on BR/FB/KA additions and
RM/RU/SW/VU), each one encoded as a trigger in TRIGGER_DEFINITIONS:
  Bowed rotor     - sharp mid-ramp resonance hump (accel-1 envelope ends
                     at ~56% of peak); peaky + one-sided accel-1 (crest
                     factor, skewness); hot at 60 Hz AND in 15<->45 ramps;
                     mic SK (5-20 kHz) high at 60 Hz and in all ramps;
                     largest 1x running-speed amplitude on accel-2/3
  Faulty bearing  - accel-1 DWT D4 (1.3-2.6 kHz) share 17-35% at low
                     speed; most accel-1 energy above 2.6 kHz (D1+D2+D3)
  Broken rotor bars - accel-1 / mic / accel-2 D4 share jumps at 60 Hz;
                     accel-1 SK (5-20 kHz) high only in ramps through 60 Hz
  Rotor misalignment - hot already at 15 Hz and flat with speed; accel-2
                     peak + mic variance hotspot at 30 Hz; largest accel-1
                     variance in 15<->45 ramps; accel-1 SK (5-20 kHz) high
                     at 45 Hz, 60 Hz, 30->60
  Rotor unbalance - accel-1 variance peaks at 45 Hz (not 60); one-sided
                     45 Hz waveform; ~+9 C only in 15<->45 ramps; least
                     impulsive (low SK); largest accel-1 1x at 45/60 Hz
  Stator winding  - 39-43 C from 30 Hz up; highest temperature variance;
                     accel-1 DC offset; vibration peaks at the start of the
                     60->30 ramp and dies within ~1 s; SK high in 60->30
  Voltage unbalance - worst in the 30->60 ramp-up on all accelerometers;
                     early burst (~5 s); heats only at 60 Hz; accel-1 SK
                     (1-5 kHz) high around 45 Hz

Built to slot directly into the existing pipeline: reuses the same
CSV-loading conventions as sensor_analysis.py / spectralKurtosis.py, so
you can call build_feature_vector() right after load_data().
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import savgol_filter
import pywt

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ======================================================================
# 1. FFT-DERIVED FEATURES
# ======================================================================
def fft_features(x, fs, harmonic_tol_hz=150, n_top_peaks=6, low_freq_cutoff_hz=300):
    """
    Quantifies:
      - low_freq_energy_frac : how much energy sits near DC
                                (large => bowed-rotor-like, broad low-freq peak)
      - harmonic_energy_ratio: how much energy sits at f0 and its first
                                few harmonics vs. total (large => healthy,
                                clean well-defined harmonics)
      - n_significant_peaks / valley_depth_ratio: shape descriptors that
                                pick up the "two peaks separated by a
                                valley" bearing-fault signature
      - spectral_centroid / spectral_flatness: general shape descriptors,
                                useful as extra classifier inputs
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    xf = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amp = (2.0 / n) * np.abs(xf)
    total_energy = np.sum(amp ** 2) or 1e-12

    low_mask = freqs <= low_freq_cutoff_hz
    low_freq_energy_frac = np.sum(amp[low_mask] ** 2) / total_energy

    # dominant peak, ignoring the true DC bin itself
    search_mask = freqs > 5
    idx_peak = int(np.argmax(amp * search_mask))
    f0 = float(freqs[idx_peak])
    a0 = float(amp[idx_peak])

    harmonic_energy = 0.0
    for k in range(1, 5):
        target = k * f0
        band = (freqs > target - harmonic_tol_hz) & (freqs < target + harmonic_tol_hz)
        harmonic_energy += np.sum(amp[band] ** 2)
    harmonic_energy_ratio = float(harmonic_energy / total_energy)

    peak_idxs, _ = signal.find_peaks(amp, distance=max(int(0.005 * n), 1))
    if len(peak_idxs) > 0:
        peak_idxs = peak_idxs[np.argsort(amp[peak_idxs])[::-1][:n_top_peaks]]
    top_peak_freqs = sorted(freqs[peak_idxs])

    valley_depth_ratio = np.nan
    if len(top_peak_freqs) >= 2:
        f_a, f_b = top_peak_freqs[0], top_peak_freqs[1]
        band = (freqs >= f_a) & (freqs <= f_b)
        if np.any(band):
            main_peak_amp = max(amp[peak_idxs[0]], 1e-12)
            valley_depth_ratio = float(np.min(amp[band]) / main_peak_amp)

    centroid = float(np.sum(freqs * amp) / (np.sum(amp) or 1e-12))
    geo_mean = np.exp(np.mean(np.log(amp + 1e-12)))
    arith_mean = np.mean(amp) + 1e-12
    spectral_flatness = float(geo_mean / arith_mean)

    return {
        "fft_f0_hz": f0,
        "fft_f0_amp": a0,
        "fft_low_freq_energy_frac": float(low_freq_energy_frac),
        "fft_harmonic_energy_ratio": harmonic_energy_ratio,
        "fft_n_significant_peaks": int(len(peak_idxs)),
        "fft_valley_depth_ratio": valley_depth_ratio,
        "fft_spectral_centroid_hz": centroid,
        "fft_spectral_flatness": spectral_flatness,
    }


# ======================================================================
# 2. DWT-DERIVED FEATURES
# ======================================================================
def dwt_band_info(fs, lev):
    """Same dyadic-halving band map used in sensor_analysis.py."""
    bands = {f"A{lev}": (0.0, fs / (2 ** (lev + 1)))}
    for k in range(1, lev + 1):
        bands[f"D{k}"] = (fs / (2 ** (k + 1)), fs / (2 ** k))
    return bands


def dwt_features(x, fs, wavelet="db4", levels=4):
    """
    Per-band energy fraction (A<lev>, D<lev> ... D1), plus:
      - dominant_band / concentration_index: which band holds the most
        energy and how concentrated it is there (bowed rotor -> A4 heavy)
      - low_band_frac: energy fraction in the lowest-frequency band
      - high_bands_frac: energy fraction in D1+D2 (healthy accel-3 signature)
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)  # remove DC/gravity-offset bias so it doesn't swamp A<lev>
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    lev = min(levels, max_level) if max_level > 0 else 1
    coeffs = pywt.wavedec(x, wavelet, level=lev)
    labels = [f"A{lev}"] + [f"D{lev - i}" for i in range(lev)]

    energies = {lab: float(np.sum(np.abs(c) ** 2)) for lab, c in zip(labels, coeffs)}
    total_energy = sum(energies.values()) or 1e-12
    fracs = {lab: e / total_energy for lab, e in energies.items()}

    dominant_band = max(fracs, key=fracs.get)
    concentration_index = fracs[dominant_band]

    out = {f"dwt_{lab}_energy_frac": frac for lab, frac in fracs.items()}
    out["dwt_dominant_band"] = dominant_band
    out["dwt_concentration_index"] = concentration_index
    out["dwt_low_band_frac"] = fracs.get(f"A{lev}", np.nan)
    out["dwt_high_bands_frac"] = fracs.get("D1", 0) + fracs.get("D2", 0)
    return out


# ======================================================================
# 3. SPECTRAL KURTOSIS FEATURES
# ======================================================================
def spectral_kurtosis_features(x, fs, window_size=2048, overlap=1024,
                                transition_hz=10000):
    """
    Quantifies the "flat then rises sharply above ~10 kHz" bowed-rotor
    signature seen on the microphone channel.
    """
    x = np.asarray(x, dtype=float)
    frequencies, _, Zxx = signal.stft(x, fs=fs, window="hann",
                                       nperseg=window_size, noverlap=overlap)
    power = np.abs(Zxx) ** 2
    sk = stats.kurtosis(power, axis=1, fisher=False, bias=False)
    sk = np.nan_to_num(sk, nan=0, posinf=0, neginf=0)
    if len(sk) > 15:
        sk = savgol_filter(sk, window_length=11, polyorder=3)

    peak_idx = int(np.argmax(sk))
    below_mask = frequencies <= transition_hz
    above_mask = ~below_mask

    mean_sk_below = float(np.mean(sk[below_mask])) if np.any(below_mask) else np.nan
    mean_sk_above = float(np.mean(sk[above_mask])) if np.any(above_mask) else np.nan

    area_below = float(_trapz(np.clip(sk[below_mask], 0, None), frequencies[below_mask])) if np.any(below_mask) else 0.0
    area_above = float(_trapz(np.clip(sk[above_mask], 0, None), frequencies[above_mask])) if np.any(above_mask) else 0.0
    total_area = (area_below + area_above) or 1e-12

    rise_ratio = (mean_sk_above / mean_sk_below) if mean_sk_below not in (0,) and not np.isnan(mean_sk_below) and mean_sk_below != 0 else np.nan

    return {
        "sk_peak_value": float(sk[peak_idx]),
        "sk_peak_freq_hz": float(frequencies[peak_idx]),
        "sk_mean_below_transition": mean_sk_below,
        "sk_mean_above_transition": mean_sk_above,
        "sk_rise_ratio": rise_ratio,
        "sk_area_frac_above_transition": area_above / total_area,
    }


# ======================================================================
# 4. GENERIC TIME-DOMAIN STATISTICS (vibration/mic channels)
# ======================================================================
def statistical_features(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    return {
        "stat_mean": float(np.mean(x)),
        "stat_median": float(np.median(x)),
        "stat_variance": float(np.var(x)),
        "stat_std": float(np.std(x)),
        "stat_rms": rms,
        "stat_peak": peak,
        "stat_crest_factor": float(peak / rms) if rms != 0 else np.nan,
        "stat_skewness": float(stats.skew(x)),
        "stat_kurtosis": float(stats.kurtosis(x)),
    }


# ======================================================================
# 5. TEMPERATURE-SPECIFIC FEATURES
# ======================================================================
def temperature_features(t, temp, warmup_frac=0.1):
    """
    peak / delta / rise_slope pick up the broken-rotor-bar ">18 C rise,
    peak near 50 C" signature. kurtosis/skewness (via statistical_features)
    pick up the bowed-rotor "highest kurtosis" and bearing "lowest
    skewness" signatures.
    """
    temp = np.asarray(temp, dtype=float)
    t = np.asarray(t, dtype=float)

    feats = statistical_features(temp)
    feats = {f"temp_{k.split('_', 1)[1]}": v for k, v in feats.items()}

    peak_temp = float(np.max(temp))
    start_temp = float(np.mean(temp[: max(1, int(len(temp) * warmup_frac))]))
    slope = float(np.polyfit(t, temp, 1)[0]) if len(t) > 1 else np.nan

    feats.update({
        "temp_peak": peak_temp,
        "temp_start": start_temp,
        "temp_delta": peak_temp - start_temp,
        "temp_rise_slope": slope,
    })
    return feats



# ======================================================================
# 5b. NEW (v2) FEATURES FOR THE EXTENDED TRIGGERS
# ======================================================================
# Speed information for the 8 test conditions. Constant-speed runs give
# the supply frequency directly; ramps give (start, end).
CONDITION_SPEEDS = {
    1: (15, 15), 2: (30, 30), 3: (45, 45), 4: (60, 60),
    5: (15, 45), 6: (30, 60), 7: (45, 15), 8: (60, 30),
}


def sk_band_features(x, fs, window_size=2048, overlap=1024, thresh=50.0):
    """
    Spectral kurtosis summarised per band, computed exactly like
    spectralKurtosis.py (Hann STFT 2048/1024, Pearson kurtosis of the
    power, Savitzky-Golay 11/3). Medians are used instead of means/peaks
    because single SK spikes are noisy run to run.
      sk_med_1_5k   : median SK, 1-5 kHz   (VU around 45 Hz, SW 60->30)
      sk_med_5_20k  : median SK, 5-20 kHz  (RM high speed, KA 60 Hz ramps,
                                            BR on the microphone)
      sk_frac_gt50  : share of 1-20 kHz bins with SK > 50 (RU lowest)
    """
    x = np.asarray(x, dtype=float)
    f, _, Z = signal.stft(x, fs=fs, window="hann", nperseg=window_size, noverlap=overlap)
    sk = stats.kurtosis(np.abs(Z) ** 2, axis=1, fisher=False, bias=False)
    sk = np.nan_to_num(sk, nan=0, posinf=0, neginf=0)
    if len(sk) > 15:
        sk = savgol_filter(sk, window_length=11, polyorder=3)
    b15 = (f >= 1000) & (f < 5000)
    b520 = (f >= 5000) & (f <= fs / 2)
    b120 = (f >= 1000) & (f <= fs / 2)
    return {
        "sk_med_1_5k": float(np.median(sk[b15])),
        "sk_med_5_20k": float(np.median(sk[b520])),
        "sk_frac_gt50": float(np.mean(sk[b120] > thresh)),
    }


def dwt_extra_features(x, fs, wavelet="db4", levels=4):
    """D4 share (1.3-2.6 kHz at fs = 41.7 kHz) and the share above D4
    (D1+D2+D3, > 2.6 kHz). Same decomposition as dwt_features()."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    coeffs = pywt.wavedec(x, wavelet, level=levels)
    e = np.array([np.sum(c ** 2) for c in coeffs])  # [A4, D4, D3, D2, D1]
    tot = e.sum() or 1e-12
    return {
        "dwt_d4_frac": float(e[1] / tot),
        "dwt_hf_frac": float(e[2:].sum() / tot),
    }


def psd_features(x, fs, nperseg=8192):
    """
    Welch power spectral density, summarised as the share of power in
    six bands plus the frequency of the PSD peak and the PSD centroid.
    (Same view as the PSD panel of the analysis plots.)
    """
    x = np.asarray(x, dtype=float)
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=min(nperseg, len(x)))
    tot = np.trapezoid(p, f) if hasattr(np, "trapezoid") else np.trapz(p, f)
    tot = tot or 1e-12
    out = {}
    for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, fs / 2)):
        m = (f >= lo) & (f < hi)
        bp = (np.trapezoid(p[m], f[m]) if hasattr(np, "trapezoid") else np.trapz(p[m], f[m])) if m.sum() > 1 else 0.0
        out[f"psd_frac_{int(lo)}_{int(min(hi, 99999))}"] = float(bp / tot)
    m = f > 5
    out["psd_peak_hz"] = float(f[m][np.argmax(p[m])])
    out["psd_centroid_hz"] = float(np.sum(f * p) / (np.sum(p) or 1e-12))
    return out


def harmonic_features(x, fs, supply_hz):
    """
    Running-speed harmonics from the full-length FFT (constant-speed runs
    only). The shaft runs slightly below the supply frequency (slip), so
    the shaft speed is found by testing candidate speeds between 0.93x and
    1.0x the supply frequency and picking the one whose first 6 harmonics
    carry the most amplitude (a "harmonic comb").
      h1_amp / h2_amp : amplitude at 1x / 2x shaft speed (m/s^2 or V)
      shaft_hz        : estimated shaft speed
    """
    if supply_hz is None:
        return {"shaft_hz": np.nan, "h1_amp": np.nan, "h2_amp": np.nan}
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    amp = np.abs(np.fft.rfft(x * np.hanning(n))) * 4.0 / n
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    df = freqs[1] - freqs[0]

    def a_at(fc, bw=0.6):
        m = (freqs > fc - bw) & (freqs < fc + bw)
        return float(amp[m].max()) if np.any(m) else 0.0

    cands = np.arange(0.93 * supply_hz, 1.0 * supply_hz + df, df)
    comb = [sum(a_at(k * c, bw=df * 1.5) for k in range(1, 7)) for c in cands]
    fr = float(cands[int(np.argmax(comb))])
    # refine on the strongest harmonic so 1x/2x land on the real peaks
    k_best = int(np.argmax([a_at(k * fr, 1.0) for k in range(1, 7)])) + 1
    m = (freqs > k_best * fr - 1.0) & (freqs < k_best * fr + 1.0)
    fr = float(freqs[m][np.argmax(amp[m])] / k_best)
    return {"shaft_hz": fr, "h1_amp": a_at(fr), "h2_amp": a_at(2 * fr)}


def envelope_features(x, fs, block_s=0.025, smooth_blocks=40):
    """
    Shape of the vibration envelope over the 10 s record -- quantifies
    the ramp observations:
      env_peak_time_frac : where the envelope peaks (0 = start, 1 = end).
                           VU's 30->60 burst at ~0.5, BR/FB/RM at ~0.7.
      env_end_over_peak  : last 10% of record vs peak. BR ~0.56 in 30->60
                           (sharp resonance hump), KA ~0.94.
      env_start_over_peak: first 10% vs peak (ramp-downs).
      env_width75_frac   : fraction of the record the envelope stays above
                           75% of its peak. SW 60->30: ~0.06-0.09.
    Envelope = peak-to-peak in 25 ms blocks, then a 1 s moving average
    (similar to reading the outline of the time-domain plot).
    """
    x = np.asarray(x, dtype=float)
    nb = max(int(block_s * fs), 1)
    nblk = len(x) // nb
    if nblk < 4:
        return {k: np.nan for k in ["env_peak_time_frac", "env_end_over_peak",
                                    "env_start_over_peak", "env_width75_frac"]}
    blocks = x[: nblk * nb].reshape(nblk, nb)
    env = blocks.max(axis=1) - blocks.min(axis=1)
    k = min(smooth_blocks, nblk)
    sm = np.convolve(env, np.ones(k) / k, mode="valid")
    pk = sm.max() or 1e-12
    n10 = max(len(sm) // 10, 1)
    return {
        "env_peak_time_frac": float((np.argmax(sm) + k / 2) / nblk),
        "env_end_over_peak": float(sm[-n10:].mean() / pk),
        "env_start_over_peak": float(sm[:n10].mean() / pk),
        "env_width75_frac": float(np.mean(sm > 0.75 * pk)),
    }

# ======================================================================
# 6. COMBINE INTO ONE FEATURE VECTOR
# ======================================================================
def build_feature_vector(df, fs, channel_map=None, temp_col_hint="temp", condition_id=None):
    """
    df           : DataFrame with a 'Time' column plus sensor columns,
                   exactly what load_data() in sensor_analysis.py /
                   spectralKurtosis.py already returns.
    fs           : sampling rate in Hz (estimate_sample_rate(df['Time']))
    channel_map  : optional dict of {friendly_name: actual_column_name},
                   e.g. {"accel1": "Accelerometer 1", "accel3": "Accelerometer 3",
                         "mic": "Microphone", "temp": "Temperature"}.
                   Used so the rule-based profiles below (which reference
                   specific accelerometers per the slides) can find the
                   right column regardless of how your CSV headers read.
                   If omitted, raw column names are used as the prefix.
    condition_id : 1-8 run/speed condition (see CONDITION_SPEEDS). Needed
                   for the running-speed harmonic features (constant-speed
                   runs 1-4); everything else works without it.

    Returns a flat dict: {"<label>__<feature_name>": value, ...} — one
    row of a feature table. Stack these (e.g. one row per test file /
    per one of the 8 speed conditions) into a DataFrame to train a
    real classifier (random forest, SVM, etc.) once you have enough
    labeled runs.
    """
    t = df["Time"].values
    numeric_cols = [c for c in df.columns if c != "Time" and pd.api.types.is_numeric_dtype(df[c])]

    col_to_label = {v: k for k, v in channel_map.items()} if channel_map else {}

    vector = {}
    for col in numeric_cols:
        label = col_to_label.get(col, col)
        x = df[col].values.astype(float)
        is_temp = (temp_col_hint in col.lower()) or (temp_col_hint in label.lower())

        if is_temp:
            feats = temperature_features(t, x)
        else:
            feats = {}
            feats.update(statistical_features(x))
            feats.update(fft_features(x, fs))
            feats.update(dwt_features(x, fs))
            feats.update(spectral_kurtosis_features(x, fs))
            # v2 trigger features
            feats.update(sk_band_features(x, fs))
            feats.update(dwt_extra_features(x, fs))
            feats.update(envelope_features(x, fs))
            feats.update(psd_features(x, fs))
            sp = CONDITION_SPEEDS.get(condition_id)
            supply = sp[0] if (sp and sp[0] == sp[1]) else None
            feats.update(harmonic_features(x, fs, supply))

        for k, v in feats.items():
            vector[f"{label}__{k}"] = v

    return vector


# ======================================================================
# 7. RULE-BASED "DISTANCE TO PROFILE" SCORER (cold-start baseline)
# ======================================================================
# Expected ranges pulled directly from the numbers quoted in the slides,
# corrected against real 8-condition data (see audit notes inline).
# `channel_map` keys (accel1 / accel3 / mic / temp) must match the labels
# used when you called build_feature_vector(). Each entry is
# (feature_name, (lo, hi), weight).
#
# IMPORTANT CAVEAT FOUND DURING VALIDATION: accel1 variance, the DWT
# low/high-band fractions, and temperature peak/delta all shift
# substantially with motor SPEED, independent of fault. These absolute
# ranges were fit across all 8 speed conditions at once, so they are a
# reasonable cold-start when you have no healthy reference at all, but
# score_fault_relative() below (which compares an unknown run against a
# healthy run at the SAME speed) is more reliable and is what
# identify_fault.py uses in practice.
FAULT_PROFILES = {
    "healthy": [
        ("accel3__dwt_high_bands_frac", (0.50, 1.00), 1.0),   # D1+D2 >60% for 6/8 conditions; widened to avoid penalizing the other 2
        ("accel1__stat_variance", (0, 30), 2.0),              # explicit healthy baseline (slide 10), widened slightly
    ],
    "bowed_rotor": [
        ("accel1__dwt_low_band_frac", (0.85, 1.00), 1.5),     # A4 >96% for 7/8 conditions; widened for the 15Hz outlier (89%)
        ("mic__sk_peak_value", (240, 450), 2.0),              # peak SK, not mean-above-transition; 350-400 typical under variable speed, lower (but still elevated vs healthy) at constant speed
        ("temp__temp_kurtosis", (5, 15), 1.0),                # "highest kurtosis" -- confirmed at every matched speed
        ("temp__temp_peak", (35, 60), 1.0),                   # exceeds 35 C at max speed
    ],
    "faulty_bearing": [
        ("accel1__stat_variance", (100, 5000), 2.0),          # >> healthy's 0-25; strongest, most reliable literal signal (confirmed 7x-100x at every speed)
        ("temp__temp_skewness", (-1.00, 0.05), 0.5),          # lowest skewness on average, not monotonic per-condition -- kept as a weak/secondary signal
    ],
    "broken_rotor_bars": [
        ("accel1__stat_variance", (1000, 8000), 2.0),         # ~2x bearing case at max constant speed -- confirmed (2.6x)
        ("temp__temp_peak", (45, 55), 1.0),                   # close to 50 C -- but ONLY seen in the 15->45 / 45->15 Hz conditions; flat elsewhere
    ],
}


def _in_range_score(value, rng):
    """1.0 inside the range, decaying toward 0 the further outside it is."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    lo, hi = rng
    if lo <= value <= hi:
        return 1.0
    span = (hi - lo) or 1.0
    dist = min(abs(value - lo), abs(value - hi))
    return max(0.0, 1.0 - dist / (span + 1e-9))


def score_fault_profiles(vector, profiles=FAULT_PROFILES):
    """
    Weighted-average (0-1) match score of each profile-defining feature.
    This is a transparent, no-training-data-needed sanity check —
    not a substitute for a classifier trained on labeled runs across
    all 8 speed conditions, but a reasonable way to narrow things down
    (or to auto-label a first batch of training data).

    Returns (scores_dict, best_guess_label).
    """
    scores = {}
    for fault, feature_specs in profiles.items():
        weighted_sum, weight_total = 0.0, 0.0
        for feat, rng, weight in feature_specs:
            weighted_sum += _in_range_score(vector.get(feat), rng) * weight
            weight_total += weight
        scores[fault] = float(weighted_sum / weight_total) if weight_total else 0.0
    best = max(scores, key=scores.get)
    return scores, best


# ======================================================================
# 7b. BASELINE-RELATIVE SCORER (recommended over the absolute version)
# ======================================================================
# The audit against real 8-condition data showed that accel1 variance,
# DWT band fractions, and temperature peak all shift with motor SPEED
# almost as much as they shift with fault. score_fault_profiles() above
# has to use one-size-fits-all ranges across all speeds and is therefore
# a blunt instrument. If you have a healthy run at the SAME speed
# condition to compare against (you do, once you've built
# healthy_feature_table.csv), comparing deltas/ratios against that
# specific baseline is a much sharper test.
def score_fault_relative(vector, healthy_baseline, in_15_45hz_window):
    """
    vector              : build_feature_vector() output for the unknown run
    healthy_baseline    : a healthy feature vector (e.g. one row of
                          healthy_feature_table.csv) from the SAME speed
                          condition -- or an average across conditions if
                          you don't know the speed, at some cost to accuracy
    in_15_45hz_window   : True if this run's speed profile is 15->45 Hz or
                          45->15 Hz (the only window where the broken-rotor-
                          bars thermal spike was observed)

    Returns (scores_dict, evidence_dict, best_guess_label).
    """
    def g(d, k, default=np.nan):
        try:
            return float(d[k])
        except (KeyError, TypeError, ValueError):
            return default

    var_ratio = g(vector, "accel1__stat_variance") / max(g(healthy_baseline, "accel1__stat_variance", 1e-9), 1e-9)
    temp_rise = g(vector, "temp__temp_peak") - g(healthy_baseline, "temp__temp_peak")
    sk_peak = g(vector, "mic__sk_peak_value")
    a4_delta = g(vector, "accel1__dwt_low_band_frac") - g(healthy_baseline, "accel1__dwt_low_band_frac")

    scores = {
        "healthy": max(0.0, 1 - abs(var_ratio - 1) / 3) * max(0.0, 1 - abs(temp_rise) / 5),
        "bowed_rotor": (
            (min(1.0, max(0.0, sk_peak) / 380) * 0.6 + min(1.0, max(0.0, a4_delta) / 0.3) * 0.4)
            * (0.4 if (in_15_45hz_window and temp_rise > 8) else 1.0)  # a big 15-45Hz temp spike points to broken rotor bars, not this
        ),
        "faulty_bearing": min(1.0, max(0.0, var_ratio - 1) / 50)
                           * (0.5 if (in_15_45hz_window and temp_rise > 8) else 1.0),
        "broken_rotor_bars": (
            0.75 * min(1.0, max(0.0, temp_rise) / 17) + 0.25 * min(1.0, max(0.0, var_ratio - 1) / 100)
        ) if in_15_45hz_window else 0.5 * min(1.0, max(0.0, var_ratio - 1) / 100),
    }
    evidence = {
        "variance_ratio_vs_healthy": var_ratio,
        "temp_peak_rise_vs_healthy_C": temp_rise,
        "mic_sk_peak_value": sk_peak,
        "dwt_a4_frac_delta_vs_healthy": a4_delta,
        "in_15_45Hz_window": in_15_45hz_window,
    }
    best = max(scores, key=scores.get)
    return scores, evidence, best



# ======================================================================
# 7c. TRIGGER ENGINE (v2) -- the recommended identification path
# ======================================================================
# Two stages, run speed REQUIRED (condition_id 1-8):
#
#   Stage A  "Is it faulty?"   Compares accel-1 and microphone variance with
#            a healthy run at the same speed. Healthy reference comes from
#            healthy_feature_table.csv if you have it, otherwise from
#            HEALTHY_REFERENCE below (taken from all_files_statistics.csv).
#
#   Stage B  "Which fault?"    Every observation from the deck is a trigger:
#            (fault, feature, the run conditions where it applies). The
#            thresholds are NOT typed in by hand -- calibrate_triggers()
#            derives them from labelled feature tables, so they can be
#            re-fitted whenever you add data (e.g. the healthy runs, or a
#            second recording of each fault).
#
# How a threshold is calibrated, per trigger and per condition:
#   - take the fault's value and every other fault's value at that speed
#   - if the fault is the highest  -> trigger = "value >= midpoint(fault,
#                                      runner-up)"
#   - if the fault is the lowest   -> trigger = "value <= midpoint"
#   - if it sits in between        -> trigger = "value inside the band
#                                      between the two neighbours' midpoints"
#   - midpoints are geometric (log scale) for strictly positive features,
#     arithmetic otherwise
#   - separation = how far the fault sits from its nearest neighbour
#     (log ratio, or share of the spread for signed features). Triggers
#     with separation below MIN_SEPARATION at a condition are dropped
#     there and reported as "not separable" -- i.e. the deck observation
#     did not hold strongly enough at that speed.

HEALTHY_REFERENCE = {
    1: {"accel1__stat_variance": 2.62734, "accel1__stat_peak": 11.423, "mic__stat_variance": 0.00024, "mic__stat_peak": 0.077382, "accel2__stat_variance": 0.145526, "accel2__stat_peak": 5.65203, "accel3__stat_variance": 0.178692, "accel3__stat_peak": 20.7525, "temp__temp_mean": 28.4446, "temp__temp_peak": 30.2971, "temp__temp_variance": 0.02111, "temp__temp_kurtosis": 6.44648, "temp__temp_skewness": 0.117027},
    2: {"accel1__stat_variance": 5.10341, "accel1__stat_peak": 13.8732, "mic__stat_variance": 0.000984, "mic__stat_peak": 0.126709, "accel2__stat_variance": 3.6841, "accel2__stat_peak": 31.315, "accel3__stat_variance": 2.30791, "accel3__stat_peak": 15.723, "temp__temp_mean": 27.7863, "temp__temp_peak": 29.3434, "temp__temp_variance": 0.020816, "temp__temp_kurtosis": 5.35616, "temp__temp_skewness": 0.061298},
    3: {"accel1__stat_variance": 9.27661, "accel1__stat_peak": 15.9366, "mic__stat_variance": 0.005782, "mic__stat_peak": 0.245217, "accel2__stat_variance": 14.0606, "accel2__stat_peak": 52.5934, "accel3__stat_variance": 10.8489, "accel3__stat_peak": 27.3616, "temp__temp_mean": 27.5311, "temp__temp_peak": 29.3763, "temp__temp_variance": 0.020864, "temp__temp_kurtosis": 5.11017, "temp__temp_skewness": 0.098748},
    4: {"accel1__stat_variance": 25.6025, "accel1__stat_peak": 23.6741, "mic__stat_variance": 0.008661, "mic__stat_peak": 0.339268, "accel2__stat_variance": 9.7538, "accel2__stat_peak": 32.1654, "accel3__stat_variance": 22.5546, "accel3__stat_peak": 51.4005, "temp__temp_mean": 27.4224, "temp__temp_peak": 28.9488, "temp__temp_variance": 0.016968, "temp__temp_kurtosis": 4.51954, "temp__temp_skewness": 0.092249},
    5: {"accel1__stat_variance": 16.5094, "accel1__stat_peak": 20.4502, "mic__stat_variance": 0.003477, "mic__stat_peak": 0.242463, "accel2__stat_variance": 3.72068, "accel2__stat_peak": 23.5573, "accel3__stat_variance": 3.12578, "accel3__stat_peak": 17.4518, "temp__temp_mean": 29.5563, "temp__temp_peak": 31.1192, "temp__temp_variance": 0.020566, "temp__temp_kurtosis": 4.78756, "temp__temp_skewness": 0.078615},
    6: {"accel1__stat_variance": 16.9895, "accel1__stat_peak": 21.1594, "mic__stat_variance": 0.02284, "mic__stat_peak": 0.926917, "accel2__stat_variance": 9.0734, "accel2__stat_peak": 35.0025, "accel3__stat_variance": 13.4731, "accel3__stat_peak": 48.5634, "temp__temp_mean": 27.5786, "temp__temp_peak": 29.1132, "temp__temp_variance": 0.018037, "temp__temp_kurtosis": 5.17274, "temp__temp_skewness": 0.076976},
    7: {"accel1__stat_variance": 16.7475, "accel1__stat_peak": 17.065, "mic__stat_variance": 0.003082, "mic__stat_peak": 0.252328, "accel2__stat_variance": 3.84326, "accel2__stat_peak": 29.2194, "accel3__stat_variance": 2.77612, "accel3__stat_peak": 19.6885, "temp__temp_mean": 29.2409, "temp__temp_peak": 31.1192, "temp__temp_variance": 0.022883, "temp__temp_kurtosis": 4.44051, "temp__temp_skewness": 0.10134},
    8: {"accel1__stat_variance": 12.7295, "accel1__stat_peak": 18.1934, "mic__stat_variance": 0.013889, "mic__stat_peak": 0.662853, "accel2__stat_variance": 9.91821, "accel2__stat_peak": 41.1482, "accel3__stat_variance": 7.3141, "accel3__stat_peak": 31.8107, "temp__temp_mean": 27.7176, "temp__temp_peak": 29.179, "temp__temp_variance": 0.023479, "temp__temp_kurtosis": 3.89769, "temp__temp_skewness": 0.102647},
}

# Stage-A thresholds: a run is flagged faulty if accel-1 OR microphone
# variance is at least this many times the healthy value at the same
# speed. Healthy/fault separation in the data: accel-1 >= 2.3x (worst
# case SW at 15 Hz), microphone >= 4.4x (SW at 30->60). 2.0 leaves margin.
FAULTY_VARIANCE_RATIO = {"accel1__stat_variance": 2.0, "mic__stat_variance": 2.0}

ALL = [1, 2, 3, 4, 5, 6, 7, 8]
CONST = [1, 2, 3, 4]
# (trigger_id, fault, feature, conditions, weight, deck observation)
TRIGGER_DEFINITIONS = [
    # ---------------- bowed rotor
    ("BR_temp_kurt", "bowed_rotor", "temp__temp_kurtosis", ALL, 1.0, "highest temperature kurtosis at every speed (orig. slide 8)"),
    ("BR_mic_sk", "bowed_rotor", "mic__sk_med_5_20k", [4, 5, 6, 7, 8], 1.5, "mic SK 5-20 kHz high at 60 Hz and in ramps (orig. slide 7 + SK slide)"),
    ("BR_a2_1x", "bowed_rotor", "accel2__h1_amp", [2, 3, 4], 1.0, "largest 1x running-speed amplitude on accel-2"),
    ("BR_a3_1x", "bowed_rotor", "accel3__h1_amp", [2, 3, 4], 1.0, "largest 1x running-speed amplitude on accel-3"),
    ("BR_temp_60", "bowed_rotor", "temp__temp_mean", [4, 6, 8], 1.0, "34.5-36.2 C in runs reaching 60 Hz"),
    ("BR_crest", "bowed_rotor", "accel1__stat_crest_factor", [1, 2, 4, 6], 0.5, "peaky accel-1 (crest factor)"),
    ("BR_skew", "bowed_rotor", "accel1__stat_skewness", [2, 4, 6, 8], 0.5, "one-sided accel-1 (skewness)"),
    ("BR_hump", "bowed_rotor", "accel1__env_end_over_peak", [6], 0.75, "sharp mid-ramp resonance hump (ends at ~56-62% of peak)"),
    ("BR_a4", "bowed_rotor", "accel1__dwt_low_band_frac", ALL, 0.5, "accel-1 DWT A4 > 96% (orig. slide 6)"),
    # ---------------- faulty bearing
    ("FB_d4", "faulty_bearing", "accel1__dwt_d4_frac", ALL, 1.5, "accel-1 D4 (1.3-2.6 kHz) share 17-35% at low speed"),
    ("FB_hf", "faulty_bearing", "accel1__dwt_hf_frac", [2, 3, 4, 5, 6, 7, 8], 1.0, "most accel-1 energy above 2.6 kHz among faults"),
    ("FB_tskew", "faulty_bearing", "temp__temp_skewness", ALL, 0.5, "lowest temperature skewness (orig. slide 11)"),
    # ---------------- broken rotor bars
    ("KA_temp", "broken_rotor_bars", "temp__temp_peak", [5, 7], 2.0, "peak ~48 C in 15<->45 ramps (orig. slide 13)"),
    ("KA_var60", "broken_rotor_bars", "accel1__stat_variance", [4], 1.0, "largest accel-1 variance at 60 Hz (orig. slide 14)"),
    ("KA_a1_d4", "broken_rotor_bars", "accel1__dwt_d4_frac", [4, 6, 8], 1.0, "accel-1 D4 share jumps at 60 Hz"),
    ("KA_mic_d4", "broken_rotor_bars", "mic__dwt_d4_frac", [4], 1.0, "mic D4 share 12.4% at 60 Hz"),
    ("KA_a2_d4", "broken_rotor_bars", "accel2__dwt_d4_frac", [4], 0.5, "accel-2 D4 share highest at 60 Hz"),
    ("KA_sk", "broken_rotor_bars", "accel1__sk_med_5_20k", [6, 8], 1.0, "accel-1 SK 5-20 kHz high only in ramps through 60 Hz"),
    # ---------------- rotor misalignment
    ("RM_temp", "rotor_misalignment", "temp__temp_mean", ALL, 1.0, "hot already at 15 Hz, flat 30.6-33.7 C"),
    ("RM_a2pk30", "rotor_misalignment", "accel2__stat_peak", [2], 1.0, "accel-2 peak highest at 30 Hz"),
    ("RM_micvar30", "rotor_misalignment", "mic__stat_variance", [2], 1.0, "mic variance 2x next fault at 30 Hz"),
    ("RM_var_ramp", "rotor_misalignment", "accel1__stat_variance", [5, 7], 1.0, "largest accel-1 variance in 15<->45 ramps"),
    ("RM_sk", "rotor_misalignment", "accel1__sk_med_5_20k", [3, 4, 6], 1.0, "accel-1 SK 5-20 kHz high at 45, 60, 30->60"),
    ("RM_skfrac", "rotor_misalignment", "accel1__sk_frac_gt50", [3, 4, 6], 0.5, "41-57% of 1-20 kHz with SK > 50"),
    # ---------------- rotor unbalance
    ("RU_var45", "rotor_unbalance", "accel1__stat_variance", [3], 1.0, "accel-1 variance peaks at 45 Hz"),
    ("RU_skew45", "rotor_unbalance", "accel1__stat_skewness", [3], 0.5, "one-sided 45 Hz waveform"),
    ("RU_temp", "rotor_unbalance", "temp__temp_peak", [5, 7], 1.5, "~38 C (+9 C) only in 15<->45 ramps"),
    ("RU_1x", "rotor_unbalance", "accel1__h1_amp", [3, 4], 1.0, "largest accel-1 1x at 45 / 60 Hz"),
    ("RU_lowsk", "rotor_unbalance", "accel1__sk_frac_gt50", ALL, 0.5, "least impulsive fault (SK > 50 in ~2% of spectrum)"),
    # ---------------- stator winding
    ("SW_temp", "stator_winding", "temp__temp_mean", ALL, 1.5, "39-43 C from 30 Hz up"),
    ("SW_tvar", "stator_winding", "temp__temp_variance", ALL, 1.0, "highest temperature variance"),
    ("SW_dc", "stator_winding", "accel1__stat_mean", ALL, 0.75, "accel-1 DC offset 6.4-9.4 m/s^2"),
    ("SW_lowvar", "stator_winding", "accel1__stat_variance", ALL, 0.5, "smallest accel-1 variance increase of all faults"),
    ("SW_burst", "stator_winding", "accel1__env_width75_frac", [8], 1.0, "60->30: vibration peaks at start, dies within ~1 s"),
    ("SW_sk_a1", "stator_winding", "accel1__sk_med_1_5k", [8], 1.0, "60->30: highest SK on accel-1"),
    ("SW_sk_a2", "stator_winding", "accel2__sk_med_1_5k", [8], 0.5, "60->30: highest SK on accel-2"),
    ("SW_sk_a3", "stator_winding", "accel3__sk_med_1_5k", [8], 0.5, "60->30: highest SK on accel-3"),
    ("SW_sk_1545", "stator_winding", "accel1__sk_med_5_20k", [5], 1.0, "15->45: highest accel-1 SK 5-20 kHz"),
    # ---------------- voltage unbalance
    ("VU_pk", "voltage_unbalance", "accel1__stat_peak", [6], 1.0, "30->60: highest accel-1 peak"),
    ("VU_v1", "voltage_unbalance", "accel1__stat_variance", [6], 0.75, "30->60: highest accel-1 variance"),
    ("VU_v2", "voltage_unbalance", "accel2__stat_variance", [6], 0.75, "30->60: highest accel-2 variance"),
    ("VU_v3", "voltage_unbalance", "accel3__stat_variance", [6], 0.75, "30->60: highest accel-3 variance"),
    ("VU_early", "voltage_unbalance", "accel1__env_peak_time_frac", [6], 1.0, "30->60: burst ~5 s in (vs ~7 s)"),
    ("VU_sk45", "voltage_unbalance", "accel1__sk_med_1_5k", [3, 5], 1.0, "accel-1 SK 1-5 kHz high around 45 Hz"),
    ("VU_temp60", "voltage_unbalance", "temp__temp_mean", [4, 6, 8], 0.75, "heats only at 60 Hz (~32 C)"),
]

MIN_SEPARATION = 0.10       # log-ratio (~10%) for positive features
MIN_SEPARATION_SIGNED = 0.05  # share of the cross-fault spread for signed features
# Features judged on an absolute scale instead: (unit, minimum gap).
# A 2 C difference matters even though it is only ~7% of 30 C.
ABSOLUTE_SCALE_FEATURES = {
    "temp__temp_mean": 1.0,   # deg C
    "temp__temp_peak": 1.0,   # deg C
    "temp__rise_vs_healthy": 1.0,   # deg C above the healthy motor at the same speed and load
}
# Minimum value a trigger needs before it may fire at all (the >7 C rule).
TRIGGER_FLOOR = {}


def _positive(vals):
    return all(v > 0 for v in vals)


def _mid(a, b, log):
    return float(np.sqrt(a * b)) if log else float((a + b) / 2)


def calibrate_triggers(feature_table, definitions=TRIGGER_DEFINITIONS,
                       min_sep=MIN_SEPARATION, min_sep_signed=MIN_SEPARATION_SIGNED):
    """
    feature_table : DataFrame of feature rows with 'fault' and
                    'condition_id' columns (all *_feature_table.csv stacked).
                    Healthy rows, if present, are ignored here (Stage A
                    handles healthy vs faulty).
    Returns a dict  {trigger_id: {..., "by_condition": {c: rule}}}
    where each rule is {"op": ">="|"<="|"between", "lo":..., "hi":...,
    "value":..., "separation":..., "usable": bool, "neighbour": fault}.
    """
    ft = feature_table[feature_table["fault"] != "healthy"]
    out = {}
    for tid, fault, feat, conds, w, note in definitions:
        rec = {"fault": fault, "feature": feat, "weight": w, "note": note, "by_condition": {}}
        for c in conds:
            sub = ft[ft["condition_id"] == c][["fault", feat]].dropna()
            if fault not in set(sub["fault"]) or len(sub) < 2:
                continue
            # one value per fault (mean if several recordings)
            vals = sub.groupby("fault")[feat].mean()
            v = float(vals[fault])
            others = vals.drop(fault)
            absolute = feat in ABSOLUTE_SCALE_FEATURES
            log = (not absolute) and _positive(list(vals.values))
            tr = (lambda z: np.log(z)) if log else (lambda z: z)
            spread = float(vals.max() - vals.min()) or 1e-12
            above = others[others > v]
            below = others[others < v]
            if len(above) == 0:
                nb = below.idxmax(); thr = _mid(v, float(below.max()), log)
                rule = {"op": ">=", "lo": thr, "hi": None}
                gap = tr(v) - tr(float(below.max()))
            elif len(below) == 0:
                nb = above.idxmin(); thr = _mid(v, float(above.min()), log)
                rule = {"op": "<=", "lo": None, "hi": thr}
                gap = tr(float(above.min())) - tr(v)
            else:
                lo_n, hi_n = float(below.max()), float(above.min())
                rule = {"op": "between", "lo": _mid(lo_n, v, log), "hi": _mid(v, hi_n, log)}
                g1, g2 = tr(v) - tr(lo_n), tr(hi_n) - tr(v)
                gap = min(g1, g2); nb = below.idxmax() if g1 <= g2 else above.idxmin()
            if absolute:
                sep = float(gap); need = ABSOLUTE_SCALE_FEATURES[feat]
            elif log:
                sep = float(gap); need = min_sep
            else:
                sep = float(gap / spread); need = min_sep_signed
            usable = bool(sep >= need)
            floor = TRIGGER_FLOOR.get(tid)
            if floor is not None:
                if v < floor:
                    usable = False
                elif rule["op"] == ">=":
                    rule["lo"] = max(rule["lo"], floor)
                elif rule["op"] == "between":
                    rule["lo"] = max(rule["lo"], floor)
                else:  # fault is the lowest but above the floor
                    rule = {"op": "between", "lo": floor, "hi": rule["hi"]}
            rule.update({"value": v, "neighbour": nb, "separation": sep, "log_scale": log,
                         "usable": usable, "floor": floor})
            rec["by_condition"][int(c)] = rule
        out[tid] = rec
    return out


def _check(rule, x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    lo, hi = rule["lo"], rule["hi"]
    if rule["op"] == ">=":
        return 1.0 if x >= lo else 0.0
    if rule["op"] == "<=":
        return 1.0 if x <= hi else 0.0
    return 1.0 if lo <= x <= hi else 0.0


def is_faulty(vector, condition_id, healthy_reference=None, ratios=FAULTY_VARIANCE_RATIO):
    """Stage A. Returns (faulty: bool, evidence dict)."""
    ref = healthy_reference if healthy_reference is not None else HEALTHY_REFERENCE.get(condition_id)
    if ref is None:
        return None, {"note": "no healthy reference for this condition"}
    ev, flags = {}, []
    for feat, k in ratios.items():
        try:
            r = float(vector[feat]) / max(float(ref[feat]), 1e-12)
        except (KeyError, TypeError, ValueError):
            continue
        ev[f"{feat}_ratio_vs_healthy"] = r
        flags.append(r >= k)
    return (any(flags) if flags else None), ev


def score_triggers(vector, condition_id, calibration, include_unusable=False):
    """
    Stage B. For each fault, adds up the weights of its triggers that fire
    at this condition.
    Returns (scores, fired, best) where
      scores[fault] = {"points": sum of fired weights,
                       "possible": sum of applicable weights,
                       "frac": points / possible}
      fired         = list of (trigger_id, fault, feature, value, rule, hit)
    Best guess = most points; ties broken by the higher fraction.
    """
    scores, fired = {}, []
    for tid, rec in calibration.items():
        rule = rec["by_condition"].get(condition_id) or rec["by_condition"].get(str(condition_id))
        if rule is None or (not rule["usable"] and not include_unusable):
            continue
        x = vector.get(rec["feature"])
        x = float(x) if x is not None else np.nan
        hit = _check(rule, x)
        s = scores.setdefault(rec["fault"], {"points": 0.0, "possible": 0.0})
        s["points"] += rec["weight"] * hit
        s["possible"] += rec["weight"]
        fired.append((tid, rec["fault"], rec["feature"], x, rule, hit))
    for s in scores.values():
        s["frac"] = s["points"] / s["possible"] if s["possible"] else 0.0
    best = max(scores, key=lambda f: (scores[f]["points"], scores[f]["frac"])) if scores else None
    if best is not None and scores[best]["points"] == 0:
        best = None
    return scores, fired, best


# ======================================================================
# 7d. TRIGGER SET v3 -- checked against the LOADED recordings (X_X_n_1)
# ======================================================================
# The loaded recordings were used as an independent check of every v2
# trigger (see validate_load.py). Outcome:
#   * temperature triggers all fail under load (every condition sits at
#     27-30 C), and a 10 s record is far too short for an 18 C rise, so
#     the unloaded temperature differences most likely reflect how long
#     the motor had been running before each recording -> all removed
#   * voltage unbalance: no trigger survives, and its cross-load matches
#     are below chance level -> class removed
#   * thresholds shift with load, so they are fitted per load
#     (calibrate_triggers on one load's table) and the healthy reference
#     must come from the same load
# Tiers:
#   "confirmed" - the pattern holds in BOTH load conditions, for a fault
#                 whose cross-load matches beat chance in a permutation
#                 test (FB p<0.01, RM p=0.02, RU p=0.04)
#   "candidate" - holds in both loads, but that fault's total number of
#                 matches is not above chance (BR, KA, SW), so each one
#                 could be a coincidence. Lower weight; confirm with
#                 repeat recordings.
HEALTHY_REFERENCE_LOADED = {
    1: {"temp__temp_mean": 27.2151, "accel1__stat_variance": 2.6737, "mic__stat_variance": 0.00112934, "accel2__stat_variance": 0.192504, "accel3__stat_variance": 0.196611},
    2: {"temp__temp_mean": 27.0291, "accel1__stat_variance": 10.3468, "mic__stat_variance": 0.00385105, "accel2__stat_variance": 2.66244, "accel3__stat_variance": 2.93145},
    3: {"temp__temp_mean": 26.9064, "accel1__stat_variance": 41.1989, "mic__stat_variance": 0.00875595, "accel2__stat_variance": 11.8899, "accel3__stat_variance": 11.833},
    4: {"temp__temp_mean": 26.8554, "accel1__stat_variance": 74.6966, "mic__stat_variance": 0.0115933, "accel2__stat_variance": 15.6927, "accel3__stat_variance": 133.532},
    5: {"temp__temp_mean": 27.8431, "accel1__stat_variance": 20.1412, "mic__stat_variance": 0.0163434, "accel2__stat_variance": 3.01048, "accel3__stat_variance": 3.56895},
    6: {"temp__temp_mean": 26.9176, "accel1__stat_variance": 51.2984, "mic__stat_variance": 0.0358334, "accel2__stat_variance": 8.36024, "accel3__stat_variance": 25.3772},
    7: {"temp__temp_mean": 27.8680, "accel1__stat_variance": 17.6807, "mic__stat_variance": 0.0102098, "accel2__stat_variance": 2.13099, "accel3__stat_variance": 2.7561},
    8: {"temp__temp_mean": 26.8636, "accel1__stat_variance": 51.1058, "mic__stat_variance": 0.0348452, "accel2__stat_variance": 7.39706, "accel3__stat_variance": 44.493},
}

TRIGGER_DEFINITIONS_V2_UNLOADED = TRIGGER_DEFINITIONS   # kept for reference

TRIGGER_TIER = {}
_V3 = [
    # tier, id, fault, feature, conditions, weight, observation
    # ---------------- faulty bearing (confirmed)
    ("confirmed", "FB_hf", "faulty_bearing", "accel1__dwt_hf_frac", [2, 3, 4, 5, 6, 7, 8], 1.5, "most accel-1 energy above 2.6 kHz (both loads)"),
    ("confirmed", "FB_d4", "faulty_bearing", "accel1__dwt_d4_frac", [1, 2, 3], 1.5, "accel-1 D4 (1.3-2.6 kHz) share 10x other faults at 15-45 Hz"),
    ("confirmed", "FB_d1", "faulty_bearing", "accel1__dwt_D1_energy_frac", [3, 4, 6], 1.0, "highest accel-1 D1 (10-21 kHz) share"),
    # ---------------- rotor misalignment (confirmed)
    ("confirmed", "RM_sk", "rotor_misalignment", "accel1__sk_med_5_20k", [3, 4], 1.5, "accel-1 SK 5-20 kHz highest at 45 / 60 Hz"),
    ("confirmed", "RM_sk_ramp", "rotor_misalignment", "accel1__sk_med_1_5k", [7], 1.0, "accel-1 SK 1-5 kHz highest in 45->15"),
    ("confirmed", "RM_smooth", "rotor_misalignment", "accel1__dwt_hf_frac", [2], 1.0, "least accel-1 energy above 2.6 kHz at 30 Hz"),
    ("confirmed", "RM_d1low", "rotor_misalignment", "accel1__dwt_D1_energy_frac", [4, 8], 1.0, "least accel-1 D1 share at 60 Hz / 60->30"),
    # ---------------- rotor unbalance (confirmed)
    ("confirmed", "RU_var45", "rotor_unbalance", "accel1__stat_variance", [3], 1.5, "accel-1 variance peaks at 45 Hz"),
    ("confirmed", "RU_skew45", "rotor_unbalance", "accel1__stat_skewness", [3], 1.0, "one-sided accel-1 waveform at 45 Hz"),
    ("confirmed", "RU_1x", "rotor_unbalance", "accel1__h1_amp", [4], 1.0, "largest accel-1 1x at 60 Hz"),
    ("confirmed", "RU_2x30", "rotor_unbalance", "accel1__h2_amp", [2], 1.0, "largest accel-1 2x at 30 Hz"),
    ("confirmed", "RU_harm45", "rotor_unbalance", "accel3__fft_harmonic_energy_ratio", [3], 1.0, "accel-3 energy least concentrated in harmonics at 45 Hz"),
    ("confirmed", "RU_d4low", "rotor_unbalance", "accel1__dwt_d4_frac", [2], 0.5, "lowest accel-1 D4 share at 30 Hz"),
    ("confirmed", "RU_lowsk", "rotor_unbalance", "accel1__sk_frac_gt50", [1, 2], 0.5, "least impulsive accel-1 (SK > 50 rarely)"),
    # ---------------- bowed rotor (candidate)
    ("candidate", "BR_a2_1x", "bowed_rotor", "accel2__h1_amp", [2], 0.75, "largest accel-2 1x at 30 Hz"),
    ("candidate", "BR_skew", "bowed_rotor", "accel1__stat_skewness", [4], 0.75, "most one-sided accel-1 at 60 Hz"),
    ("candidate", "BR_sk60", "bowed_rotor", "accel1__sk_med_1_5k", [4], 0.75, "accel-1 SK 1-5 kHz highest at 60 Hz"),
    ("candidate", "BR_kurt15", "bowed_rotor", "accel1__stat_kurtosis", [1], 0.5, "highest accel-1 kurtosis at 15 Hz"),
    ("candidate", "BR_hump", "bowed_rotor", "accel2__env_width75_frac", [5], 0.5, "narrowest accel-2 resonance burst in 15->45"),
    # ---------------- broken rotor bars (candidate)
    ("candidate", "KA_low1x_mic", "broken_rotor_bars", "mic__h1_amp", [3, 4], 0.75, "smallest 1x in the sound at 45 / 60 Hz"),
    ("candidate", "KA_low1x_a2", "broken_rotor_bars", "accel2__h1_amp", [4], 0.5, "smallest accel-2 1x at 60 Hz"),
    ("candidate", "KA_low1x_a3", "broken_rotor_bars", "accel3__h1_amp", [3, 4], 0.5, "smallest accel-3 1x at 45 / 60 Hz"),
    ("candidate", "KA_a3var60", "broken_rotor_bars", "accel3__stat_variance", [4], 0.75, "largest accel-3 variance at 60 Hz"),
    ("candidate", "KA_a2pk60", "broken_rotor_bars", "accel2__stat_peak", [4], 0.5, "largest accel-2 peak at 60 Hz"),
    # ---------------- stator winding (candidate)
    ("candidate", "SW_kurt8", "stator_winding", "accel1__stat_kurtosis", [8], 0.75, "highest accel-1 kurtosis in 60->30"),
    ("candidate", "SW_a3pk8", "stator_winding", "accel3__stat_peak", [8], 0.75, "lowest accel-3 peak in 60->30"),
    ("candidate", "SW_lowvar45", "stator_winding", "accel1__stat_variance", [3], 0.5, "smallest accel-1 variance at 45 Hz"),
    ("candidate", "SW_a2var45", "stator_winding", "accel2__stat_variance", [3], 0.5, "smallest accel-2 variance at 45 Hz"),
    ("candidate", "SW_start", "stator_winding", "accel3__env_start_over_peak", [5], 0.5, "accel-3 strongest at the start of 15->45"),
]
TRIGGER_DEFINITIONS_V3 = []
for tier, *d in _V3:
    TRIGGER_DEFINITIONS_V3.append(tuple(d))
    TRIGGER_TIER[d[0]] = tier
V3_FAULTS = ["bowed_rotor", "faulty_bearing", "broken_rotor_bars",
             "rotor_misalignment", "rotor_unbalance", "stator_winding"]


def healthy_reference_for(condition_id, load=0):
    """Built-in healthy reference for Stage A at this speed and load."""
    return (HEALTHY_REFERENCE_LOADED if load == 1 else HEALTHY_REFERENCE).get(condition_id)


# ======================================================================
# 7e. FINAL TRIGGER GROUPS -- both loads / unloaded only / loaded only
# ======================================================================
# Every trigger belongs to one group:
#   "both"      holds in the unloaded AND the loaded recordings
#               (TRIGGER_DEFINITIONS_V3; tier confirmed or tentative)
#   "unloaded"  holds only without load: consistent inside the unloaded
#               data (5 s halves / reverse ramp) but not under load
#   "loaded"    holds only under load: found in the loaded runs and
#               consistent inside the loaded data, not seen unloaded
# A motor is scored with "both" + the group of its own load.
#
# TEMPERATURE ASSUMPTION: a run counts as a temperature trigger only if
# the motor is MORE THAN 7 C above the healthy motor at the same speed and
# load (feature temp__rise_vs_healthy). Smaller rises (e.g. the ~4-5 C seen
# for several faults in the same runs) are treated as heating from the
# test schedule, not as a fault effect. Under load no fault reaches 7 C.
#
# How the groups were checked: analysis/rule_consistency.py (rule fitted
# on the full runs must fire for its fault and no other fault in the
# independent looks), analysis/consistency_search.py (loaded search +
# chance level).
TEMP_RISE_THRESHOLD_C = 7.0

TRIGGERS_UNLOADED_ONLY = [
    ("U_KA_temp", "broken_rotor_bars", "temp__rise_vs_healthy", [5, 7], 2.0, ">7 C above healthy in the 15<->45 ramps (+15.9 / +16.8 C)"),
    ("U_SW_temp", "stator_winding", "temp__rise_vs_healthy", [2, 3, 4, 5, 6, 7, 8], 1.5, ">7 C above healthy from 30 Hz up (+10.2 to +15.2 C)"),
    ("U_BR_temp", "bowed_rotor", "temp__rise_vs_healthy", [4, 6], 1.0, ">7 C above healthy at 60 Hz / 30->60 (+7.1 / +8.6 C; reverse ramp only +6.8 C)"),
    ("U_RU_temp", "rotor_unbalance", "temp__rise_vs_healthy", [5, 7], 1.0, ">7 C above healthy in the 15<->45 ramps (+7.0 / +7.3 C)"),
    ("U_BR_a3_1x", "bowed_rotor", "accel3__h1_amp", [2, 3, 4], 1.0, "largest accel-3 1x at 30-60 Hz"),
    ("U_BR_crest", "bowed_rotor", "accel1__stat_crest_factor", [4], 0.5, "peakiest accel-1 at 60 Hz"),
    ("U_KA_var60", "broken_rotor_bars", "accel1__stat_variance", [4], 1.0, "largest accel-1 variance at 60 Hz"),
    ("U_KA_mic_d4", "broken_rotor_bars", "mic__dwt_d4_frac", [4], 1.0, "most sound energy in 1.3-2.6 kHz at 60 Hz"),
    ("U_KA_a1_d4", "broken_rotor_bars", "accel1__dwt_d4_frac", [6, 8], 1.0, "accel-1 1.3-2.6 kHz share jumps in ramps through 60 Hz"),
    ("U_RM_micvar30", "rotor_misalignment", "mic__stat_variance", [2], 1.0, "loudest microphone at 30 Hz"),
    ("U_RM_var_ramp", "rotor_misalignment", "accel1__stat_variance", [5, 7], 1.0, "largest accel-1 variance in both 15<->45 ramps"),
    ("U_SW_burst", "stator_winding", "accel1__env_width75_frac", [8], 0.5, "60->30: vibration peaks at the start and dies within ~1 s (direction-dependent, not checkable)"),
]
for _t in TRIGGERS_UNLOADED_ONLY:
    if _t[2] == "temp__rise_vs_healthy":
        TRIGGER_FLOOR[_t[0]] = TEMP_RISE_THRESHOLD_C

TRIGGERS_LOADED_ONLY = [
    ("L_BR_a1hf15", "bowed_rotor", "accel1__psd_frac_10000_20833", [1], 0.75, "most accel-1 power above 10 kHz at 15 Hz"),
    ("L_BR_a1_1x30", "bowed_rotor", "accel1__h1_amp", [2], 0.75, "smallest accel-1 1x at 30 Hz"),
    ("L_BR_a1d3_45", "bowed_rotor", "accel1__dwt_D3_energy_frac", [3], 0.5, "least accel-1 energy in 2.6-5.2 kHz at 45 Hz"),
    ("L_BR_mic2x60", "bowed_rotor", "mic__h2_amp", [4], 0.5, "smallest 2x in the sound at 60 Hz"),
    ("L_BR_micvar", "bowed_rotor", "mic__stat_variance", [5, 7], 0.75, "quietest microphone in both 15<->45 ramps"),
    ("L_BR_mic25k", "bowed_rotor", "mic__psd_frac_2000_5000", [5, 7], 0.5, "most sound power in 2-5 kHz in both 15<->45 ramps"),
    ("L_BR_micskew", "bowed_rotor", "mic__stat_skewness", [6, 8], 0.5, "most one-sided sound in both 30<->60 ramps"),
    ("L_FB_skrise", "faulty_bearing", "accel1__sk_rise_ratio", [1], 0.5, "accel-1 SK rises most above 10 kHz at 15 Hz"),
    ("L_FB_mic510", "faulty_bearing", "mic__psd_frac_5000_10000", [2, 4], 0.75, "most sound power in 5-10 kHz at 30 / 60 Hz"),
    ("L_FB_michf", "faulty_bearing", "mic__dwt_hf_frac", [3], 0.5, "most sound energy above 2.6 kHz at 45 Hz"),
    ("L_FB_a1d3", "faulty_bearing", "accel1__dwt_D3_energy_frac", [6, 8], 0.75, "most accel-1 energy in 2.6-5.2 kHz in both 30<->60 ramps"),
    ("L_KA_a2skew", "broken_rotor_bars", "accel2__stat_skewness", [2, 6], 0.5, "most one-sided accel-2 at 30 Hz and 30->60"),
    ("L_KA_a3d1", "broken_rotor_bars", "accel3__dwt_D1_energy_frac", [3], 0.5, "most accel-3 energy above 10 kHz at 45 Hz"),
    ("L_KA_a2_12k", "broken_rotor_bars", "accel2__psd_frac_1000_2000", [4], 0.75, "most accel-2 power in 1-2 kHz at 60 Hz"),
    ("L_KA_a1skew", "broken_rotor_bars", "accel1__stat_skewness", [6], 0.5, "least one-sided accel-1 in 30->60"),
    ("L_KA_micskew", "broken_rotor_bars", "mic__stat_skewness", [3], 0.5, "most negative sound skewness at 45 Hz"),
    ("L_RM_mic05", "rotor_misalignment", "mic__psd_frac_500_1000", [2], 0.75, "most sound power in 0.5-1 kHz at 30 Hz"),
    ("L_RM_a1_05", "rotor_misalignment", "accel1__psd_frac_500_1000", [3], 0.5, "most accel-1 power in 0.5-1 kHz at 45 Hz"),
    ("L_RM_micharm", "rotor_misalignment", "mic__fft_harmonic_energy_ratio", [4], 0.5, "sound least concentrated in harmonics at 60 Hz"),
    ("L_RM_crest", "rotor_misalignment", "accel1__stat_crest_factor", [5, 7], 1.0, "peakiest accel-1 in both 15<->45 ramps"),
    ("L_RM_a1kurt", "rotor_misalignment", "accel1__stat_kurtosis", [5], 0.5, "spikiest accel-1 in 15->45"),
    ("L_RM_a2kurt", "rotor_misalignment", "accel2__stat_kurtosis", [6, 8], 0.5, "spikiest accel-2 in both 30<->60 ramps"),
    ("L_RU_a2_1x15", "rotor_unbalance", "accel2__h1_amp", [1], 0.5, "largest accel-2 1x at 15 Hz"),
    ("L_RU_a3skew", "rotor_unbalance", "accel3__stat_skewness", [2, 3], 0.5, "accel-3 skewness extreme at 30 (low) / 45 Hz (high)"),
    ("L_RU_a2skpk", "rotor_unbalance", "accel2__sk_peak_value", [4], 0.75, "highest accel-2 SK peak at 60 Hz"),
    ("L_RU_a3kurt", "rotor_unbalance", "accel3__stat_kurtosis", [4], 0.5, "spikiest accel-3 at 60 Hz"),
    ("L_RU_a1hf", "rotor_unbalance", "accel1__dwt_hf_frac", [7], 0.75, "least accel-1 energy above 2.6 kHz in 45->15"),
    ("L_RU_micsk", "rotor_unbalance", "mic__sk_med_5_20k", [6, 8], 0.5, "least impulsive sound above 5 kHz in both 30<->60 ramps"),
    ("L_SW_mic05", "stator_winding", "mic__psd_frac_500_1000", [1, 2, 5, 7], 1.0, "sound power in 0.5-1 kHz: highest at 15 Hz, lowest at 30 Hz and in 15<->45"),
    ("L_SW_mic510", "stator_winding", "mic__psd_frac_5000_10000", [1, 3], 0.5, "sound power in 5-10 kHz extreme at 15 (high) / 45 Hz (low)"),
    ("L_SW_a1_25k", "stator_winding", "accel1__psd_frac_2000_5000", [4], 0.5, "least accel-1 power in 2-5 kHz at 60 Hz"),
    ("L_SW_mickurt", "stator_winding", "mic__stat_kurtosis", [6], 0.75, "least spiky sound in 30->60"),
]
TRIGGER_GROUP = {d[0]: "both" for d in TRIGGER_DEFINITIONS_V3}
TRIGGER_GROUP.update({d[0]: "unloaded" for d in TRIGGERS_UNLOADED_ONLY})
TRIGGER_GROUP.update({d[0]: "loaded" for d in TRIGGERS_LOADED_ONLY})
TRIGGER_TIER.update({d[0]: "unloaded-only" for d in TRIGGERS_UNLOADED_ONLY})
TRIGGER_TIER.update({d[0]: "loaded-only" for d in TRIGGERS_LOADED_ONLY})


def definitions_for_load(load):
    """Triggers used for a motor at this load: 'both' + that load's own group."""
    return list(TRIGGER_DEFINITIONS_V3) + list(TRIGGERS_UNLOADED_ONLY if load == 0 else TRIGGERS_LOADED_ONLY)


def add_relative_features(vector, healthy):
    """Adds features measured against the healthy motor at the same speed
    and load. healthy = a healthy feature row (dict) or None."""
    v = dict(vector)
    try:
        v["temp__rise_vs_healthy"] = float(v["temp__temp_mean"]) - float(healthy["temp__temp_mean"])
    except (KeyError, TypeError, ValueError):
        v["temp__rise_vs_healthy"] = float("nan")
    return v

# ======================================================================
# 8. OPTIONAL: TRAIN A REAL CLASSIFIER ONCE YOU HAVE LABELED RUNS
# ======================================================================
# score_fault_profiles() above is a hand-tuned, no-training-data-needed
# baseline. Its weak spot is exactly what you'd expect from fixed
# thresholds: faults that share a symptom (e.g. bowed rotor and broken
# rotor bars both run hot) can tie or flip depending on where a value
# happens to fall relative to a boundary. The real fix is to let a
# classifier learn the boundaries from your labeled runs (you already
# have 8 known conditions per fault type in the dataset) instead of
# guessing them by hand. This needs scikit-learn (`pip install
# scikit-learn`).
def train_fault_classifier(feature_rows, labels, n_estimators=200, random_state=0):
    """
    feature_rows : list of dicts (or a DataFrame) -- one build_feature_vector()
                   output per labeled run.
    labels       : list of fault-condition strings, same length/order.

    Returns (trained_pipeline, feature_columns). The pipeline imputes
    missing values, scales, and fits a random forest; feature_columns
    records the exact column order the pipeline expects at predict time.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame(feature_rows)
    # drop non-numeric columns (e.g. dwt_dominant_band, a string label)
    X = X.select_dtypes(include=[np.number])
    feature_columns = list(X.columns)

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
    ])
    pipeline.fit(X.values, labels)
    return pipeline, feature_columns


def predict_fault(pipeline, feature_columns, vector):
    """Predict a fault label (+ per-class probabilities) for one feature vector dict."""
    row = pd.DataFrame([{c: vector.get(c, np.nan) for c in feature_columns}])
    pred = pipeline.predict(row.values)[0]
    proba = dict(zip(pipeline.named_steps["clf"].classes_, pipeline.predict_proba(row.values)[0]))
    return pred, proba
