"""FFT for LPF outputs (pose/torque)."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from analyze_pushup_fft import compute_fft, dominant_peaks, plot_spectrum

FS = 30.0
POSE_DIR = Path("output_data/filtered_pose_lpf")
TORQUE_DIR = Path("output_data/filtered_torque_lpf")
OUT_DIR = Path("fft_plots")
OUT_DIR.mkdir(exist_ok=True)

POSE_TRIPLE_CANDIDATES = (
    ("joint_16_x", "joint_16_y", "joint_16_z"),
    ("wrist_R_x", "wrist_R_y", "wrist_R_z"),
    ("joint_0_x", "joint_0_y", "joint_0_z"),
)

TORQUE_COLUMN_CANDIDATES = (
    "wrist_R_local_y",
    "wrist_R_y",
    "elbow_R_local_y",
    "elbow_R_y",
)


def _select_triplet(df: pd.DataFrame):
    for triplet in POSE_TRIPLE_CANDIDATES:
        if all(c in df.columns for c in triplet):
            return df[list(triplet)].to_numpy(float)
    raise KeyError("no xyz triplet found")


def _select_column(df: pd.DataFrame):
    for col in TORQUE_COLUMN_CANDIDATES:
        if col in df.columns:
            return df[col].to_numpy(float)
    raise KeyError("no torque column found")


def _fft_mag(sig: np.ndarray):
    sig = np.nan_to_num(sig, nan=np.nanmean(sig))
    freq, amp = compute_fft(sig, FS)
    return freq, amp


def main():
    results = []

    for p in sorted(POSE_DIR.glob("*.csv")):
        df = pd.read_csv(p).interpolate(limit_direction="both")
        try:
            xyz = _select_triplet(df)
        except KeyError:
            print(f"[SKIP] pose {p.name} (no xyz)")
            continue
        mag = np.linalg.norm(xyz, axis=1)
        if np.all(~np.isfinite(mag)):
            print(f"[SKIP] pose {p.name} (all NaN)")
            continue
        freq, amp = _fft_mag(mag)
        peaks = dominant_peaks(freq, amp, top_k=3)
        plot_path = OUT_DIR / f"{p.stem}_lpf_fft.png"
        try:
            plot_spectrum(freq, amp, str(plot_path), logx=True)
        except Exception as e:
            plot_path = None
            print(f"[WARN] plot failed {p.name}: {e}")
        results.append({"file": str(p), "type": "pose_lpf", "peaks": peaks, "plot": str(plot_path) if plot_path else None})
        print(f"[POSE_LPF] {p.name} peaks={peaks}")

    for p in sorted(TORQUE_DIR.glob("*.csv")):
        df = pd.read_csv(p).interpolate(limit_direction="both")
        try:
            sig = _select_column(df)
        except KeyError:
            print(f"[SKIP] torque {p.name} (no col)")
            continue
        if np.all(~np.isfinite(sig)):
            print(f"[SKIP] torque {p.name} (all NaN)")
            continue
        freq, amp = _fft_mag(sig)
        peaks = dominant_peaks(freq, amp, top_k=3)
        plot_path = OUT_DIR / f"{p.stem}_lpf_fft.png"
        try:
            plot_spectrum(freq, amp, str(plot_path), logx=True)
        except Exception as e:
            plot_path = None
            print(f"[WARN] plot failed {p.name}: {e}")
        results.append({"file": str(p), "type": "torque_lpf", "peaks": peaks, "plot": str(plot_path) if plot_path else None})
        print(f"[TORQUE_LPF] {p.name} peaks={peaks}")

    out = OUT_DIR / "fft_peaks_lpf.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[OUT] {out}")


if __name__ == "__main__":
    main()
