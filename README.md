# Smartphone-Based Pedestrian Step Detection (Digital Signal Processing)

Step counting from smartphone IMU data. The project compares three denoising filters (**Moving Average**, **Butterworth low-pass** and **1D Kalman**) combined with **speed-adaptive peak detection**, and evaluates them on 54 walking recordings against reference step counts.

**Period:** August 2026 · **Language:** Python · **Full write-up:** [`Report.pdf`](Report.pdf)

---

## Key results

Step-count accuracy over all 54 recordings (higher is better):

| Filter | Mean accuracy | Mean absolute error (steps) | Noise reduction* |
|---|---|---|---|
| **Moving Average** | **92.5 %** | 3.26 | 71.1 % |
| **1D Kalman** | 91.1 % | 3.87 | 77.4 % |
| Butterworth (4th order, 3 Hz) | 77.5 % | 10.67 | 76.3 % |

\*Reduction in the standard deviation of the acceleration magnitude relative to the raw signal.

**Accuracy by carrying position and speed** (%):

| Position | Speed | Moving Avg | Butterworth | Kalman |
|---|---|---|---|---|
| Arm | slow | 89.7 | 91.3 | 90.7 |
| Arm | normal | 94.9 | 73.0 | 92.0 |
| Arm | fast | 94.2 | 67.1 | 92.5 |
| Hand | slow | 92.6 | 85.5 | 89.7 |
| Hand | normal | 93.3 | 80.8 | 91.7 |
| Hand | fast | 90.5 | 72.9 | 87.9 |
| Pocket | slow | 93.2 | 86.7 | 90.6 |
| Pocket | normal | 91.3 | 77.1 | 92.5 |
| Pocket | fast | 93.1 | 63.2 | 92.3 |

**Takeaways**
- Moving Average and Kalman both stay above 87 % in every position and speed combination, so they handle changes in phone placement and gait well.
- Butterworth is competitive at slow walking, but its accuracy drops to about 63–73 % at fast walking. The 3 Hz cutoff is a likely cause, since fast stepping produces a stronger content in that band than slow stepping. See the report for the discussion.
- Carrying position makes little difference to the averaged accuracy of Moving Average (92.1–92.9 %) and Kalman (89.8–91.8 %).

---

## Dataset

54 IMU recordings collected with a smartphone:

- 3 participants × 3 carrying positions (**hand**, **arm**, **pocket**) × 3 walking speeds (**slow**, **normal**, **fast**) × 2 trials
- Sampled at 100 Hz, about 18–23 s per recording (reference counts range from 28 to 58 steps)
- Each CSV holds a timestamp, 3-axis accelerometer (`ax, ay, az`) and 3-axis gyroscope (`wx, wy, wz`)
- File naming: `Participant <n>_<position>_<speed>_Trial <k>.csv`
- Reference step counts live in `PTB data/ground_truth.csv`

---

## Method

1. **Cleaning:** parse timestamps, drop invalid and duplicate rows, estimate the sampling rate and flag timing gaps over 2 s.
2. **Feature:** compute the acceleration magnitude `√(ax² + ay² + az²)`, which is independent of phone orientation.
3. **Filtering** of the magnitude signal:
   - **Moving Average:** centered window of about 0.12 s
   - **Butterworth:** zero-phase (`filtfilt`) 4th-order low-pass at 3 Hz
   - **Kalman:** 1D discrete filter (process variance 1e-3, measurement variance 1e-1)
4. **Adaptive peak detection:** `scipy.signal.find_peaks` with a minimum peak distance and a prominence threshold that depend on the **filter** and the **walking speed** (slow, normal, fast).
5. **Parameter tuning:** a grid search over minimum gap (0.15–0.45 s) and prominence factor (0.05–0.50) for each filter and speed.
6. **Evaluation:** accuracy `= max(0, 100 − |detected − reference| / reference × 100)`, averaged per filter, position and speed.

**Tuned parameters used** (minimum gap in s, prominence factor):

| Filter | Slow | Normal | Fast |
|---|---|---|---|
| Moving Average | 0.45, 0.10 | 0.30, 0.10 | 0.25, 0.10 |
| Butterworth | 0.35, 0.20 | 0.15, 0.10 | 0.20, 0.05 |
| Kalman | 0.40, 0.30 | 0.30, 0.15 | 0.25, 0.10 |

---

## Repository structure

```
code/
├── Python DSP.py              # Full pipeline: clean → filter → detect → evaluate → plot
├── Report.pdf                 # Project report
├── PTB data/
│   ├── Original/              # 54 raw IMU recordings (CSV)
│   ├── Cleaned/               # Generated: cleaned signals + result tables
│   │   ├── *_clean.csv                         # Per-recording filtered signals
│   │   ├── summary_all_recordings.csv          # Step counts per filter
│   │   ├── step_detection_accuracy_evaluation.csv
│   │   ├── accuracy_summary_by_placement.csv
│   │   ├── signal_noise_reduction.csv
│   │   └── parameter_tuning_results.csv        # Grid-search results
│   └── ground_truth.csv       # Reference step count per recording
└── plots/                     # Generated figures (PNG)
```

---

## Getting started

**Requirements:** Python 3.9+ with `numpy`, `pandas`, `scipy` and `matplotlib`.

```bash
pip install numpy pandas scipy matplotlib
```

**Run** from inside the `code/` folder. The script uses relative paths:

```bash
cd code
python "Python DSP.py"
```

It processes all recordings, writes the cleaned data and result tables to `PTB data/Cleaned/`, runs the parameter grid search, and saves the figures to `plots/`. The grid search re-reads every cleaned file for each parameter combination, so expect it to take a few minutes.

---

## Figures

| Figure | File |
|---|---|
| Raw vs filtered signals | `plots/final_filter_comparison.png` |
| Detected steps for all three filters | `plots/final_step_detection_all_filters.png` |
| Accuracy by position and speed | `plots/final_accuracy_by_placement_and_speed.png` |
| Effect of carrying position | `plots/comparison_position_Participant_2_normal.png` |
| Effect of walking speed | `plots/comparison_speed_Participant_1_hand.png` |

---

## Limitations

- The detection parameters were tuned and evaluated on the **same 54 recordings** with no held-out test set, so the reported accuracies are likely somewhat optimistic. Cross-validation, for example leave-one-participant-out, would give a fairer estimate.
- Only 3 participants were recorded, and the results may not generalize to other gaits, phones or surfaces.
- The Kalman filter uses fixed noise variances, and the Butterworth cutoff is fixed at 3 Hz. Neither was tuned per condition.
- Accuracy is measured on the total step count per recording, not on the timing of individual steps.

---

## Possible extensions

- Per-participant or per-condition tuning of the Butterworth cutoff and Kalman variances
- Using the gyroscope magnitude as a second channel for peak detection
- Held-out or leave-one-participant-out evaluation
- Real-time (causal) implementation on a device
