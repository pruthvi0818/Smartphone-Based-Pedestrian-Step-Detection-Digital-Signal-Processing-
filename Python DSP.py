import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, filtfilt


# 1. Directory Configurations

RAW = Path("PTB data/Original")
FIG = Path("plots")
PREP = Path("PTB data/Cleaned")

FIG.mkdir(parents=True, exist_ok=True)
PREP.mkdir(parents=True, exist_ok=True)

FILENAME_PATTERN = re.compile(r"Participant\s*(\d+)_(hand|arm|pocket)_(slow|normal|fast)_Trial\s*(\d+)\.csv")



# 2. Data Cleaning & Loading Function

def load_and_clean(path):
    """Read one CSV, normalize timestamps, compute magnitudes, and filter."""
    df = pd.read_csv(path, comment="#")
    df = df.rename(columns={"wx": "gx", "wy": "gy", "wz": "gz"})
    df = df[["time", "ax", "ay", "az", "gx", "gy", "gz"]]

    # Timestamp conversion
    time_text = df["time"].astype(str).str.replace("Z", "", regex=False)
    time_as_number = pd.to_numeric(time_text, errors="coerce")

    if time_as_number.notna().mean() > 0.9:
        seconds = time_as_number - time_as_number.iloc[0]
    else:
        has_ampm = time_text.str.contains("AM|PM")
        real_time_12h = pd.to_datetime(time_text, format="%Y-%m-%dT%I:%M:%S.%f %p", errors="coerce")
        real_time_24h = pd.to_datetime(time_text, format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce")
        real_time = real_time_12h.where(has_ampm, real_time_24h)
        seconds = (real_time - real_time.iloc[0]).dt.total_seconds()

    df["time"] = seconds

    # Clean numeric data
    sensor_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["time"] + sensor_cols).drop_duplicates(subset="time")
    df = df.sort_values("time").reset_index(drop=True)

    # Compute sampling frequency
    time_steps = np.diff(df["time"])
    time_steps = time_steps[time_steps > 0]
    fs = 1 / np.median(time_steps) if len(time_steps) else np.nan
    max_gap = time_steps.max() if len(time_steps) else np.nan

    # Calculate 3D Vector Magnitudes
    acc_matrix = df[["ax", "ay", "az"]].to_numpy()
    gyro_matrix = df[["gx", "gy", "gz"]].to_numpy()

    df["acc_mag"] = np.linalg.norm(acc_matrix, axis=1)
    df["gyro_mag"] = np.linalg.norm(gyro_matrix, axis=1)

    return df, fs, max_gap



# 3. Filter Algorithms & Speed-Adaptive Peak Detection

def moving_average_filter(signal, window_size=15):
    """Moving Average Smoothing Filter."""
    return pd.Series(signal).rolling(window=max(3, window_size), center=True).mean().bfill().ffill().to_numpy()


def butterworth_lowpass_filter(signal, fs, cutoff_hz=3.0, order=4):
    """Zero-Phase 4th Order Butterworth Low-Pass Filter."""
    if np.isnan(fs) or fs <= 0:
        fs = 50.0
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def kalman_filter_1d(signal, process_variance=1e-3, measurement_variance=1e-1):
    """1D Discrete Kalman Filter for Dynamic State Estimation."""
    n = len(signal)
    xhat, P = np.zeros(n), np.zeros(n)
    xhatminus, Pminus, K = np.zeros(n), np.zeros(n), np.zeros(n)

    xhat[0], P[0] = signal[0], 1.0

    for k in range(1, n):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + process_variance
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (signal[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


def find_steps_filter(signal, fs, speed="normal", min_gap_sec=None, prom_factor=None, filter_name="generic"):
    """Adaptive Peak Detection scaling with walking speed dynamic thresholds."""

    TUNED_PARAMS = {
        ("MovingAvg", "slow"): (0.45, 0.10),
        ("MovingAvg", "normal"): (0.30, 0.10),
        ("MovingAvg", "fast"): (0.25, 0.10),
        ("Butterworth", "slow"): (0.35, 0.20),
        ("Butterworth", "normal"): (0.15, 0.10),
        ("Butterworth", "fast"): (0.20, 0.05),
        ("Kalman", "slow"): (0.40, 0.30),
        ("Kalman", "normal"): (0.30, 0.15),
        ("Kalman", "fast"): (0.25, 0.10),

    }

    if min_gap_sec is None or prom_factor is None:
        min_gap_sec, prom_factor = TUNED_PARAMS.get(
            (filter_name, speed), (0.30, 0.25)  # fallback if not found
        )

    scale = 2.0 if filter_name == "Butterworth" else 5.0

    min_gap = max(1, int(min_gap_sec * fs)) if fs and not np.isnan(fs) else 3
    peak_prominence = max(0.10, prom_factor * np.std(signal) * scale)
    peaks, _ = find_peaks(signal, distance=min_gap, prominence=peak_prominence)
    return peaks


# 4. Processing 54 Files & Exporting Summaries

summary_rows = []
gap_warnings = []

for file_path in sorted(RAW.glob("*.csv")):
    match = FILENAME_PATTERN.match(file_path.name)
    if not match:
        print(f"  SKIPPED (no regex match): {file_path.name}")
        continue
    participant_num, position, speed, trial = match.groups()
    participant = f"P{participant_num}"   # matches ground_truth.csv style

    data, fs_i, max_gap = load_and_clean(file_path)

    # Calculate filtered signals
    data["acc_ma"] = moving_average_filter(data["acc_mag"], window_size=int(0.12 * fs_i))
    data["acc_bw"] = butterworth_lowpass_filter(data["acc_mag"], fs_i, cutoff_hz=3.0)
    data["acc_km"] = kalman_filter_1d(data["acc_mag"])

    # Speed-adaptive step detection per filter
    peaks_ma = find_steps_filter(data["acc_ma"], fs_i, speed=speed, filter_name="MovingAvg")
    peaks_bw = find_steps_filter(data["acc_bw"], fs_i, speed=speed, filter_name="Butterworth")
    peaks_km = find_steps_filter(data["acc_km"], fs_i, speed=speed, filter_name="Kalman")

    if max_gap and max_gap > 2.0:
        gap_warnings.append((file_path.name, round(max_gap, 2)))

    summary_rows.append({
        "trial_id": file_path.stem,
        "participant": participant,
        "position": position,
        "speed": speed,
        "trial": int(trial),
        "duration_s": round(data["time"].iloc[-1], 1),
        "samples": len(data),
        "fs_hz": round(fs_i, 1),
        "peaks_MovingAvg": len(peaks_ma),
        "peaks_Butterworth": len(peaks_bw),
        "peaks_Kalman": len(peaks_km),
    })

    # Export cleaned dataset
    clean_out = PREP / f"{file_path.stem}_clean.csv"
    data.to_csv(clean_out, index=False)

if not summary_rows:
    raise RuntimeError(f"No files matched FILENAME_PATTERN in {RAW}. Check filenames against the regex.")

summary_df = pd.DataFrame(summary_rows).sort_values(["participant", "position", "speed", "trial"])
summary_df.to_csv(PREP / "summary_all_recordings.csv", index=False)
summary_df.to_csv(PREP / "filter_benchmark_comparison.csv", index=False)

print(f"\n[✓] Successfully processed {len(summary_df)} recordings.")
print("Recordings with timing gaps above 2s:", gap_warnings)


# 4b. Signal Quality Evaluation (Noise Reduction)

noise_rows = []
for file_path in sorted(PREP.glob("*_clean.csv")):
    c_df = pd.read_csv(file_path)
    noise_rows.append({
        "trial_id": file_path.stem.replace("_clean", ""),
        "std_raw": c_df["acc_mag"].std(),
        "std_MovingAvg": c_df["acc_ma"].std(),
        "std_Butterworth": c_df["acc_bw"].std(),
        "std_Kalman": c_df["acc_km"].std(),
    })

noise_df = pd.DataFrame(noise_rows)
noise_df.to_csv(PREP / "signal_noise_reduction.csv", index=False)

mean_raw = noise_df["std_raw"].mean()
print("\n=== SIGNAL QUALITY (NOISE REDUCTION) ===")
print(f"Mean raw std: {mean_raw:.4f} m/s^2")
for col in ["std_MovingAvg", "std_Butterworth", "std_Kalman"]:
    mean_val = noise_df[col].mean()
    reduction = (1 - mean_val / mean_raw) * 100
    print(f"{col}: mean = {mean_val:.4f} m/s^2, reduction = {reduction:.1f}%")



# 5. Ground Truth Loading & Evaluation

df_bench = summary_df.copy()
gt_csv = Path("PTB data/ground_truth.csv")

if gt_csv.exists():
    gt_data = pd.read_csv(gt_csv)
    if "ground_truth" in gt_data.columns:
        gt_subset = gt_data[["trial_id", "ground_truth"]]
        df_bench = df_bench.merge(gt_subset, on="trial_id", how="left")

# Compute Accuracy and Absolute Error
for col in ["peaks_MovingAvg", "peaks_Butterworth", "peaks_Kalman"]:
    filter_name = col.replace("peaks_", "")
    df_bench[f"error_{filter_name}"] = np.abs(df_bench[col] - df_bench["ground_truth"])
    df_bench[f"accuracy_{filter_name}"] = np.maximum(
        0, 100 - ((df_bench[f"error_{filter_name}"] / df_bench["ground_truth"]) * 100)
    )

df_bench.to_csv(PREP / "step_detection_accuracy_evaluation.csv", index=False)
df_eval = df_bench.dropna(subset=["ground_truth"])

print("\n=== STEP DETECTION ACCURACY EVALUATION (%) ===")
if not df_eval.empty:
    acc_summary = df_eval.groupby(["position", "speed"])[
        ["accuracy_MovingAvg", "accuracy_Butterworth", "accuracy_Kalman"]
    ].mean().round(2)
    print(acc_summary)
    acc_summary.to_csv(PREP / "accuracy_summary_by_placement.csv")



# 6. Dynamic Speed Parameter Fine-Tuning Grid Search

print("\n=== PARAMETER FINE-TUNING GRID SEARCH ===")
tuning_results = []

if not df_eval.empty:
    for f_col, f_name in [("acc_ma", "MovingAvg"), ("acc_bw", "Butterworth"), ("acc_km", "Kalman")]:
        for speed_cat in ["slow", "normal", "fast"]:
            for min_gap in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                for prom_factor in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
                    errors, accuracies = [], []
                    for idx, row in df_eval[df_eval["speed"] == speed_cat].iterrows():
                        clean_file = PREP / f"{row['trial_id']}_clean.csv"
                        if not clean_file.exists():
                            continue
                        c_df = pd.read_csv(clean_file)
                        p_det = find_steps_filter(
                            c_df[f_col], row["fs_hz"], speed=speed_cat,
                            min_gap_sec=min_gap, prom_factor=prom_factor, filter_name=f_name
                        )
                        err = abs(len(p_det) - row["ground_truth"])
                        acc = max(0, 100 - (err / row["ground_truth"] * 100))
                        errors.append(err);
                        accuracies.append(acc)

                    if errors:
                        tuning_results.append({
                            "filter": f_name, "speed": speed_cat,
                            "min_gap_sec": min_gap, "prom_factor": prom_factor,
                            "mean_abs_error": round(np.mean(errors), 2),
                            "mean_accuracy_pct": round(np.mean(accuracies), 2)
                        })

    tuning_df = pd.DataFrame(tuning_results).sort_values("mean_abs_error")
    tuning_df.to_csv(PREP / "parameter_tuning_results.csv", index=False)

    print("\nTop 5 Parameter Configurations (Across All Filters):")
    print(tuning_df.head(5).to_string(index=False))



# 7a. Visualizations

example_file = RAW / "Participant 1_arm_fast_Trial 1.csv"
if example_file.exists():
    df_ex, fs_ex, _ = load_and_clean(example_file)
    raw_sig = df_ex["acc_mag"].to_numpy()
    ma_sig = moving_average_filter(raw_sig, window_size=int(0.12 * fs_ex))
    bw_sig = butterworth_lowpass_filter(raw_sig, fs_ex, cutoff_hz=3.0)
    km_sig = kalman_filter_1d(raw_sig)

    plt.figure(figsize=(11, 5))
    plt.plot(df_ex["time"], raw_sig, color="lightgray", label="Raw Acceleration Magnitude", alpha=0.7)
    plt.plot(df_ex["time"], ma_sig, label="Moving Average", linewidth=1.5)
    plt.plot(df_ex["time"], bw_sig, label="Butterworth Low-Pass (3Hz)", linewidth=1.5)
    plt.plot(df_ex["time"], km_sig, label="1D Kalman Filter", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Filter Comparison: Raw vs MA vs Butterworth (3Hz) vs Kalman")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "filter_comparison_all_methods.png", dpi=300, bbox_inches="tight")
    plt.close()

if not df_eval.empty:
    acc_summary.plot(kind="bar", figsize=(11, 5))
    plt.title("Step Detection Accuracy by Placement and Speed (Speed-Adaptive Dynamic Thresholds)")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Position, Speed")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "accuracy_by_placement_and_speed.png", dpi=300)
    plt.close()


    # ----------------------------------------------------------------------------
    # 7b. Additional Diagnostic Figures (positions, speeds, individual filters,
    #     and the pre-tuning baseline that motivated speed-adaptive thresholds)
    # ----------------------------------------------------------------------------
    def find_steps_fixed(signal, fs, min_gap_sec=0.30, prom_factor=0.5):
        """Original fixed-threshold detection, used only here to illustrate
        the under-counting problem that motivated speed-adaptive tuning."""
        min_gap = max(1, int(min_gap_sec * fs)) if fs and not np.isnan(fs) else 3
        peak_prominence = max(0.10, prom_factor * np.std(signal))
        peaks, _ = find_peaks(signal, distance=min_gap, prominence=peak_prominence)
        return peaks


    def load_trial(fname):
        d, f, _ = load_and_clean(RAW / fname)
        d["acc_ma"] = moving_average_filter(d["acc_mag"], window_size=int(0.12 * f))
        d["acc_bw"] = butterworth_lowpass_filter(d["acc_mag"], f, cutoff_hz=3.0)
        d["acc_km"] = kalman_filter_1d(d["acc_mag"])
        return d, f


    if example_file.exists():
        row1 = df_bench[df_bench["trial_id"] == "Participant 1_arm_fast_Trial 1"].iloc[0]

        # Raw axes / magnitude / gyroscope
        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], df_ex["ax"], label="X", lw=0.8)
        plt.plot(df_ex["time"], df_ex["ay"], label="Y", lw=0.8)
        plt.plot(df_ex["time"], df_ex["az"], label="Z", lw=0.8)
        plt.title("Raw Acc Data (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration");
        plt.legend(loc="upper right");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_axes.png", dpi=200);
        plt.close()

        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], raw_sig, label="Acceleration magnitude", lw=0.8)
        plt.title("Magnitude in Acceleration  (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration");
        plt.legend(loc="upper left");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_acc_mag.png", dpi=200);
        plt.close()

        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], df_ex["gyro_mag"], label="Gyroscope magnitude", lw=0.8)
        plt.title("Magnitude in Gyroscope (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Angular velocity");
        plt.legend(loc="upper left");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_gyro_mag.png", dpi=200);
        plt.close()

        # Raw vs Moving Average
        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], raw_sig, label="Raw", lw=0.8, color="tab:blue")
        plt.plot(df_ex["time"], ma_sig, label="Moving average", lw=1.6, color="tab:orange")
        plt.title("Raw vs Smoothed Signal (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration");
        plt.legend(loc="upper right");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_smoothing.png", dpi=200);
        plt.close()

        # Butterworth / Kalman isolated
        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], bw_sig, label="Butterworth Low-Pass Filter", color="tab:orange", lw=1.4)
        plt.title("Filtering Technique Comparison (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration (m/s^2)");
        plt.legend(loc="upper right");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_butterworth_isolated.png", dpi=200);
        plt.close()

        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], km_sig, label="1D Kalman Filter", color="tab:green", lw=1.4)
        plt.title("Filtering Technique Comparison (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration (m/s^2)");
        plt.legend(loc="upper right");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_kalman_isolated.png", dpi=200);
        plt.close()

        # Filter comparison demo (duplicate style of filter_comparison_all_methods.png)
        plt.figure(figsize=(11, 5.5))
        plt.plot(df_ex["time"], raw_sig, color="lightgray", alpha=0.7, label="Raw Acceleration Magnitude")
        plt.plot(df_ex["time"], ma_sig, label="Moving Average Filter", lw=1.4)
        plt.plot(df_ex["time"], bw_sig, label="Butterworth Low-Pass Filter (3Hz)", lw=1.4)
        plt.plot(df_ex["time"], km_sig, label="1D Kalman Filter", lw=1.4)
        plt.title("Filtering Technique Comparison (Participant 1, Arm, Fast, Trial 1)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration (m/s^2)");
        plt.legend(loc="lower right");
        plt.tight_layout()
        plt.savefig(FIG / "filter_comparison_demo.png", dpi=200);
        plt.close()

        # Pre-tuning baseline (fixed threshold) — illustrates the under-counting problem
        peaks_fixed_bw = find_steps_fixed(bw_sig, fs_ex)
        peaks_fixed_ma = find_steps_fixed(ma_sig, fs_ex)

        plt.figure(figsize=(11, 4.5))
        plt.plot(df_ex["time"], raw_sig, color="lightgray", alpha=0.6, label="Raw Acc Magnitude")
        plt.plot(df_ex["time"], bw_sig, color="tab:blue", label="Butterworth Low-Pass (3 Hz)")
        plt.scatter(df_ex["time"].to_numpy()[peaks_fixed_bw], bw_sig[peaks_fixed_bw],
                    color="red", zorder=5, label=f"Detected Peaks ({len(peaks_fixed_bw)})")
        plt.title(
            f"Butterworth Low-Pass Filter & Step Peaks - Participant 1 Arm Fast Trial 1 ({len(peaks_fixed_bw)} steps)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration (m/s^2)");
        plt.legend(loc="upper right");
        plt.tight_layout()
        plt.savefig(FIG / "Participant_1_arm_fast_Trial_1_peaks.png", dpi=200);
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.plot(df_ex["time"], ma_sig, "--", color="lightblue", label="Moving Average Signal")
        plt.plot(df_ex["time"], bw_sig, color="navy", label="Butterworth Signal")
        plt.scatter(df_ex["time"].to_numpy()[peaks_fixed_bw], bw_sig[peaks_fixed_bw],
                    color="red", zorder=5, label=f"Butterworth Peaks ({len(peaks_fixed_bw)})")
        plt.scatter(df_ex["time"].to_numpy()[peaks_fixed_ma], ma_sig[peaks_fixed_ma],
                    color="black", marker="x", zorder=5, label=f"Moving Avg Peaks ({len(peaks_fixed_ma)})")
        plt.title(
            f"Step Detection Accuracy: Butterworth ({len(peaks_fixed_bw)} steps) vs Moving Avg ({len(peaks_fixed_ma)} steps)")
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration (m/s^2)");
        plt.legend(loc="upper right");
        plt.tight_layout()
        plt.savefig(FIG / "step_detection_filter_impact.png", dpi=200);
        plt.close()

        # Final tuned detection (per-filter, matching Table 3 parameters)
        p_ma = find_steps_filter(ma_sig, fs_ex, speed="fast", filter_name="MovingAvg")
        p_bw = find_steps_filter(bw_sig, fs_ex, speed="fast", filter_name="Butterworth")
        p_km = find_steps_filter(km_sig, fs_ex, speed="fast", filter_name="Kalman")

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, (sig, peaks, name, color) in zip(
                axes,
                [(ma_sig, p_ma, "Moving Average", "tab:blue"),
                 (bw_sig, p_bw, "Butterworth (3Hz)", "tab:orange"),
                 (km_sig, p_km, "1D Kalman Filter", "tab:green")]
        ):
            ax.plot(df_ex["time"], raw_sig, color="lightgray", alpha=0.6, label="Raw")
            ax.plot(df_ex["time"], sig, color=color, linewidth=1.3, label=name)
            ax.scatter(df_ex["time"].to_numpy()[peaks], sig[peaks], color="red", zorder=5,
                       label=f"Detected ({len(peaks)})")
            ax.set_ylabel("Acc (m/s^2)");
            ax.legend(loc="upper right");
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_title(f"{name}: {len(peaks)} steps detected (ground truth = {int(row1['ground_truth'])})")
        axes[-1].set_xlabel("Time (s)")
        plt.suptitle("Step Detection with Final Tuned Parameters (Participant 1, Arm, Fast, Trial 1)", y=1.01)
        plt.tight_layout()
        plt.savefig(FIG / "final_step_detection_all_filters.png", dpi=200, bbox_inches="tight");
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.plot(df_ex["time"], raw_sig, color="lightgray", label="Raw Acceleration Magnitude", alpha=0.7)
        plt.plot(df_ex["time"], ma_sig, label="Moving Average", linewidth=1.5)
        plt.plot(df_ex["time"], bw_sig, label="Butterworth Low-Pass (3Hz)", linewidth=1.5)
        plt.plot(df_ex["time"], km_sig, label="1D Kalman Filter", linewidth=1.5)
        plt.xlabel("Time (s)");
        plt.ylabel("Acceleration (m/s\u00b2)")
        plt.title("Filter Comparison: Raw vs MA vs Butterworth vs Kalman (Participant 1, Arm, Fast, Trial 1)")
        plt.legend(loc="upper right");
        plt.grid(True, linestyle="--", alpha=0.5);
        plt.tight_layout()
        plt.savefig(FIG / "final_filter_comparison.png", dpi=200, bbox_inches="tight");
        plt.close()

    if not df_eval.empty:
        acc_summary.plot(kind="bar", figsize=(11, 5))
        plt.title("Step Detection Accuracy by Placement and Speed (Final Tuned Parameters, n=54)")
        plt.ylabel("Accuracy (%)");
        plt.xlabel("Position, Speed");
        plt.ylim(0, 105)
        plt.grid(True, linestyle="--", alpha=0.5);
        plt.tight_layout()
        plt.savefig(FIG / "final_accuracy_by_placement_and_speed.png", dpi=200);
        plt.close()

    # Position comparison: Participant 2, normal speed
    dfh, _ = load_trial("Participant 2_hand_normal_Trial 1.csv")
    dfa, _ = load_trial("Participant 2_arm_normal_Trial 1.csv")
    dfp, _ = load_trial("Participant 2_pocket_normal_Trial 1.csv")

    plt.figure(figsize=(11, 4.5))
    plt.plot(dfh["time"], dfh["acc_mag"], label="hand", lw=0.7)
    plt.plot(dfa["time"], dfa["acc_mag"], label="arm", lw=0.7)
    plt.plot(dfp["time"], dfp["acc_mag"], label="pocket", lw=0.7)
    plt.title("Comparison in Position  - Participant 2, Normal speed (Hand vs Arm vs Pocket)")
    plt.xlabel("Time (s)");
    plt.ylabel("Acceleration");
    plt.legend(loc="upper right");
    plt.tight_layout()
    plt.savefig(FIG / "comparison_position_Participant_2_normal.png", dpi=200);
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (d_, name) in zip(axes, [(dfh, "HAND"), (dfa, "ARM"), (dfp, "POCKET")]):
        ax.plot(d_["time"], d_["acc_mag"], color="lightgray", alpha=0.6, label="Raw Signal")
        ax.plot(d_["time"], d_["acc_bw"], label="Butterworth (3Hz)", lw=1.2)
        ax.plot(d_["time"], d_["acc_km"], label="Kalman Filter", lw=1.2)
        ax.set_title(f"Position Impact: {name} (Participant 2, Normal Speed, Trial 1)")
        ax.set_ylabel("Acc (m/s^2)");
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(FIG / "filter_comparison_positions.png", dpi=200);
    plt.close()

    # Speed comparison: Participant 1, hand position
    df_slow, _ = load_trial("Participant 1_hand_slow_Trial 1.csv")
    df_norm, _ = load_trial("Participant 1_hand_normal_Trial 1.csv")
    df_fast, _ = load_trial("Participant 1_hand_fast_Trial 1.csv")

    plt.figure(figsize=(11, 4.5))
    plt.plot(df_slow["time"], df_slow["acc_mag"], label="slow", lw=0.7)
    plt.plot(df_norm["time"], df_norm["acc_mag"], label="normal", lw=0.7)
    plt.plot(df_fast["time"], df_fast["acc_mag"], label="fast", lw=0.7)
    plt.title("Comparison in Speed - Participant 1, Hand (Slow vs Normal vs Fast)")
    plt.xlabel("Time (s)");
    plt.ylabel("Acceleration");
    plt.legend(loc="lower right");
    plt.tight_layout()
    plt.savefig(FIG / "comparison_speed_Participant_1_hand.png", dpi=200);
    plt.close()

    print(f"[\u2713] Additional diagnostic figures saved. Total figures in '{FIG}':", len(list(FIG.glob("*.png"))))

print(f"\n[✓] Process complete. Updated outputs saved to '{PREP}' and '{FIG}'.")