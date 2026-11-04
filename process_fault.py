"""
process_fault.py
=================
Generic version of process_healthy.py: point it at a glob pattern +
fault label and it builds a feature table the same way, using the
corrected fault_feature_extraction pipeline.

Usage:
    python3 process_fault.py <glob_pattern> <fault_label> <output_csv>

Example:
    python3 process_fault.py "B_R_*_0.csv" bowed_rotor bowed_rotor_feature_table.csv

Batch mode (v2) -- build one table per fault folder in one go:
    python3 process_fault.py --all <data_root> <out_dir> [load]
  where <data_root> holds folders named BR, FB, HH, KA, RM, RU, SW, VU
  (any that exist are processed). Writes <out_dir>/<label>_feature_table.csv.
  load = 0 (unloaded, files X_X_n_0.csv, default) or 1 (loaded, X_X_n_1.csv).

If TIME_CSV is not found, the time axis is rebuilt from FALLBACK_FS so the
scripts still run on machines without the time file.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

from fault_feature_extraction import build_feature_vector

# Path to the time CSV, relative to wherever you run this script from.
# If your folder layout differs, just change this one line.
TIME_CSV = "1_Time/Time_10_seconds_420000_samples.csv"
FALLBACK_FS = 1.0 / 2.4e-5   # 41,666.7 Hz -- what the time CSV gives

FAULT_LABELS = {
    "HH": "healthy", "BR": "bowed_rotor", "FB": "faulty_bearing",
    "KA": "broken_rotor_bars", "RM": "rotor_misalignment",
    "RU": "rotor_unbalance", "SW": "stator_winding", "VU": "voltage_unbalance",
}


def _condition_id(path):
    """Extract the condition number (1-8) from a filename like
    H_H_3_0.csv, regardless of what folder it's sitting in
    (BR\\B_R_3_0.csv, ./FB/F_B_3_0.csv, etc.)."""
    return int(os.path.basename(path).split("_")[2])

def _load_id(path):
    """Last number in the filename: 0 = unloaded, 1 = loaded (X_X_3_1.csv)."""
    try:
        return int(os.path.basename(path).split("_")[3].split(".")[0])
    except (IndexError, ValueError):
        return None


SPEED_LABELS = {
    1: "15 Hz (const)",
    2: "30 Hz (const)",
    3: "45 Hz (const)",
    4: "60 Hz (const)",
    5: "15->45 Hz (accel)",
    6: "30->60 Hz (accel)",
    7: "45->15 Hz (decel)",
    8: "60->30 Hz (decel)",
}

CHANNEL_MAP = {
    "accel1": "Accelerometer 1 (m/s^2)",
    "accel2": "Accelerometer 2 (m/s^2)",
    "accel3": "Accelerometer 3 (m/s^2)",
    "mic": "Microphone (V)",
    "temp": "Temperature (Celsius)",
}


def load_data(time_csv, sensor_csv):
    s_df = pd.read_csv(sensor_csv, encoding="utf-8-sig")
    if not os.path.exists(time_csv):
        s_df = s_df.loc[:, ~s_df.columns.str.contains("^Unnamed")].dropna(how="all").reset_index(drop=True)
        df = s_df.copy()
        df.insert(0, "Time", np.arange(len(df)) / FALLBACK_FS)
        return df
    t_df = pd.read_csv(time_csv, encoding="utf-8-sig")

    t_df = t_df.loc[:, ~t_df.columns.str.contains("^Unnamed")]
    s_df = s_df.loc[:, ~s_df.columns.str.contains("^Unnamed")]

    time_col = [c for c in t_df.columns if "time" in c.lower()][0]
    t_df = t_df.dropna(subset=[time_col]).reset_index(drop=True)
    s_df = s_df.dropna(how="all").reset_index(drop=True)

    n = min(len(t_df), len(s_df))
    if len(t_df) != len(s_df):
        print(f"  Warning: row mismatch (time={len(t_df)}, sensors={len(s_df)}). Truncating to {n}.")

    df = s_df.iloc[:n].copy()
    df.insert(0, "Time", t_df[time_col].iloc[:n].values)
    return df


def estimate_sample_rate(time_values):
    dt = np.diff(time_values)
    dt = dt[dt > 0]
    return 1.0 / np.median(dt)


def process(glob_pattern, fault_label, out_path):
    files = sorted(glob.glob(glob_pattern), key=_condition_id)
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pattern!r}")

    rows = []
    for f in files:
        cond_num = _condition_id(f)
        print(f"Processing {f} (condition {cond_num}: {SPEED_LABELS.get(cond_num, '?')}) ...")

        df = load_data(TIME_CSV, f)
        fs = estimate_sample_rate(df["Time"].values)
        print(f"  estimated fs = {fs:,.2f} Hz")

        vec = build_feature_vector(df, fs, channel_map=CHANNEL_MAP, condition_id=cond_num)
        vec["fault"] = fault_label
        vec["condition_id"] = cond_num
        vec["load"] = _load_id(f)
        vec["speed_profile"] = SPEED_LABELS.get(cond_num, "unknown")
        vec["source_file"] = f
        rows.append(vec)

    feat_df = pd.DataFrame(rows)
    id_cols = ["source_file", "fault", "condition_id", "load", "speed_profile"]
    other_cols = [c for c in feat_df.columns if c not in id_cols]
    feat_df = feat_df[id_cols + other_cols]

    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved feature table: {out_path}  ({feat_df.shape[0]} rows x {feat_df.shape[1]} cols)")
    return feat_df


def process_all(data_root, out_dir, load=0):
    os.makedirs(out_dir, exist_ok=True)
    tables = []
    for code, label in FAULT_LABELS.items():
        pat = os.path.join(data_root, code, f"{code[0]}_{code[1]}_*_{load}.csv")
        if not glob.glob(pat):
            print(f"(no files for {code}, skipping)")
            continue
        tables.append(process(pat, label, os.path.join(out_dir, f"{label}_feature_table.csv")))
    return pd.concat(tables, ignore_index=True) if tables else None


if __name__ == "__main__":
    if sys.argv[1] == "--all":
        # optional 4th argument: load suffix (0 = unloaded, 1 = loaded)
        process_all(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        sys.exit(0)
    pattern, fault_label, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    feat_df = process(pattern, fault_label, out_path)

    headline_cols = [
        "condition_id", "speed_profile",
        "accel1__stat_variance", "accel1__fft_low_freq_energy_frac",
        "accel1__dwt_low_band_frac", "accel3__dwt_high_bands_frac",
        "mic__sk_mean_below_transition", "mic__sk_mean_above_transition",
        "temp__temp_peak", "temp__temp_delta", "temp__temp_kurtosis", "temp__temp_skewness",
        # v2 trigger features
        "temp__temp_mean", "temp__temp_variance", "accel1__dwt_d4_frac", "accel1__dwt_hf_frac",
        "accel1__sk_med_1_5k", "accel1__sk_med_5_20k", "mic__sk_med_5_20k",
        "accel1__h1_amp", "accel2__h1_amp", "accel1__env_peak_time_frac", "accel1__env_width75_frac",
    ]
    print(f"\n=== Headline features: {fault_label} ===")
    print(feat_df[headline_cols].to_string(index=False))