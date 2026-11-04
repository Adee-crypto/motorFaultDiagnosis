import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import fault_feature_extraction as F
V3KEYS = {(d[1], d[2]) for d in F.TRIGGER_DEFINITIONS_V3}
TEMP = [
    ("U_KA_temp", "broken_rotor_bars", "temp__rise_vs_healthy", [5, 7], 2.0, ">7 C above healthy in the 15<->45 ramps (+15.9 to +16.8 C)"),
    ("U_SW_temp", "stator_winding", "temp__rise_vs_healthy", [2, 3, 4, 5, 6, 7, 8], 1.5, ">7 C above healthy from 30 Hz up (+10.2 to +15.2 C)"),
    ("U_BR_temp", "bowed_rotor", "temp__rise_vs_healthy", [4, 6], 1.0, ">7 C above healthy at 60 Hz and 30->60 (+7.1, +8.6 C)"),
    ("U_RU_temp", "rotor_unbalance", "temp__rise_vs_healthy", [5, 7], 1.0, ">7 C above healthy in the 15<->45 ramps (+7.0, +7.3 C)"),
]
for t in TEMP:
    F.TRIGGER_FLOOR[t[0]] = 7.0
UNLOADED_CAND = [("U_" + d[0],) + tuple(d[1:]) for d in F.TRIGGER_DEFINITIONS
                 if d[1] != "voltage_unbalance" and not d[2].startswith("temp__") and (d[1], d[2]) not in V3KEYS] + TEMP
LOADED_CAND = [
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
    ("L_KA_a2skew", "broken_rotor_bars", "accel2__stat_skewness", [2, 5, 6, 7], 0.5, "most one-sided accel-2 (positive skew)"),
    ("L_KA_a3d1", "broken_rotor_bars", "accel3__dwt_D1_energy_frac", [3], 0.5, "most accel-3 energy above 10 kHz at 45 Hz"),
    ("L_KA_a2_12k", "broken_rotor_bars", "accel2__psd_frac_1000_2000", [4], 0.75, "most accel-2 power in 1-2 kHz at 60 Hz"),
    ("L_KA_a1skew", "broken_rotor_bars", "accel1__stat_skewness", [6, 8], 0.5, "least one-sided accel-1 in both 30<->60 ramps"),
    ("L_KA_micskew", "broken_rotor_bars", "mic__stat_skewness", [3], 0.5, "most negative sound skewness at 45 Hz"),
    ("L_RM_mic05", "rotor_misalignment", "mic__psd_frac_500_1000", [2], 0.75, "most sound power in 0.5-1 kHz at 30 Hz"),
    ("L_RM_a1_05", "rotor_misalignment", "accel1__psd_frac_500_1000", [3], 0.5, "most accel-1 power in 0.5-1 kHz at 45 Hz"),
    ("L_RM_micharm", "rotor_misalignment", "mic__fft_harmonic_energy_ratio", [4], 0.5, "sound least concentrated in harmonics at 60 Hz"),
    ("L_RM_crest", "rotor_misalignment", "accel1__stat_crest_factor", [5, 7], 1.0, "peakiest accel-1 (crest factor) in both 15<->45 ramps"),
    ("L_RM_a1kurt", "rotor_misalignment", "accel1__stat_kurtosis", [5], 0.5, "spikiest accel-1 in 15->45"),
    ("L_RM_a2kurt", "rotor_misalignment", "accel2__stat_kurtosis", [6, 8], 0.5, "spikiest accel-2 in both 30<->60 ramps"),
    ("L_RU_a2_1x15", "rotor_unbalance", "accel2__h1_amp", [1], 0.5, "largest accel-2 1x at 15 Hz"),
    ("L_RU_a3skew", "rotor_unbalance", "accel3__stat_skewness", [2, 3], 0.5, "accel-3 skewness extreme at 30 (low) / 45 Hz (high)"),
    ("L_RU_a2skpk", "rotor_unbalance", "accel2__sk_peak_value", [4], 0.75, "highest accel-2 SK peak at 60 Hz"),
    ("L_RU_a3kurt", "rotor_unbalance", "accel3__stat_kurtosis", [4], 0.5, "spikiest accel-3 at 60 Hz"),
    ("L_RU_a1hf", "rotor_unbalance", "accel1__dwt_hf_frac", [5, 7], 0.75, "least accel-1 energy above 2.6 kHz in both 15<->45 ramps"),
    ("L_RU_micsk", "rotor_unbalance", "mic__sk_med_5_20k", [6, 8], 0.5, "least impulsive sound above 5 kHz in both 30<->60 ramps"),
    ("L_SW_mic05", "stator_winding", "mic__psd_frac_500_1000", [1, 2, 5, 7], 1.0, "sound power in 0.5-1 kHz: highest at 15 Hz, lowest at 30 Hz and in 15<->45"),
    ("L_SW_mic510", "stator_winding", "mic__psd_frac_5000_10000", [1, 3], 0.5, "sound power in 5-10 kHz extreme at 15 (high) / 45 Hz (low)"),
    ("L_SW_a1_25k", "stator_winding", "accel1__psd_frac_2000_5000", [4], 0.5, "least accel-1 power in 2-5 kHz at 60 Hz"),
    ("L_SW_mickurt", "stator_winding", "mic__stat_kurtosis", [6, 8], 0.75, "least spiky sound in both 30<->60 ramps"),
]
