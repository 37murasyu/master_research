"""Batch FFT for pose (2–9) and torque outputs.

Outputs dominant peaks (top 3) for each file and saves spectra plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analyze_pushup_fft import compute_fft, dominant_peaks, plot_spectrum


FS = 30.0


POSE_FILES: Sequence[str] = (
    "Adjusted 3D Pose/2_stereo_pose.csv",
    "Adjusted 3D Pose/3_0stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/3_1stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/4_0stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/5_1stereo_pose_scaled.csv",
    "Adjusted 3D Pose/5_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/6_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/7_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/8_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/kpts3d_9_20250925_201442.csv",
)

POSE_TRIPLE_CANDIDATES: Sequence[Sequence[str]] = (
    ("joint_16_x", "joint_16_y", "joint_16_z"),
    ("wrist_R_x", "wrist_R_y", "wrist_R_z"),
    ("joint_0_x", "joint_0_y", "joint_0_z"),
)

TORQUE_FILES: Sequence[str] = (
    "torque/2_stereo_pose_torque.csv",
    "torque/3_0stereo_pose_scaled_with2d_torque.csv",
    "torque/3_1stereo_pose_scaled_with2d_torque.csv",
    "torque/4_0stereo_pose_scaled_with2d_torque.csv",
    "torque/5_1stereo_pose_scaled_torque.csv",
    "torque/5_stereo_pose_scaled_with2d_torque.csv",
    "torque/6_stereo_pose_scaled_with2d_torque.csv",
    "torque/7_stereo_pose_scaled_with2d_torque.csv",
    "torque/8_stereo_pose_scaled_with2d_torque.csv",
    "torque/kpts3d_9_20250925_201442_torque.csv",
)

TORQUE_COLUMN_CANDIDATES: Sequence[str] = (
    "wrist_R_local_y",
    "wrist_R_y",
    "elbow_R_local_y",
    "elbow_R_y",
)


def _select_triplet(df: pd.DataFrame, candidates: Iterable[Sequence[str]]) -> np.ndarray:
    for triplet in candidates:
        if all(c in df.columns for c in triplet):
            return df[list(triplet)].to_numpy(float)
    raise KeyError("no xyz triplet found")


def _select_column(df: pd.DataFrame, candidates: Iterable[str]) -> np.ndarray:
    for col in candidates:
        if col in df.columns:
            return df[col].to_numpy(float)
    raise KeyError("no torque column found")


def _fft_mag(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.nan_to_num(sig, nan=np.nanmean(sig))
    freq, amp = compute_fft(arr, fs)
    return freq, amp


def main() -> None:
    results: list[dict] = []
    plot_dir = Path("fft_plots")
    plot_dir.mkdir(exist_ok=True)

    for path in POSE_FILES:
        p = Path(path)
        if not p.exists():
            print(f"[MISS] {path}")
            continue
        df = pd.read_csv(p)
        try:
            xyz = _select_triplet(df, POSE_TRIPLE_CANDIDATES)
        except KeyError:
            print(f"[SKIP] {path} (no xyz triplet)")
            continue
        mag = np.linalg.norm(xyz, axis=1)
        freq, amp = _fft_mag(mag, FS)
        peaks = dominant_peaks(freq, amp, top_k=3)
        plot_path = plot_dir / f"{p.stem}_fft.png"
        try:
            plot_spectrum(freq, amp, str(plot_path), logx=True)
        except Exception as e:  # plotting may fail if matplotlib missing
            plot_path = None
            print(f"[WARN] plot failed for {path}: {e}")
        results.append({"file": path, "type": "pose", "peaks": peaks, "plot": str(plot_path) if plot_path else None})
        print(f"[POSE] {path} peaks={peaks} plot={plot_path}")

    for path in TORQUE_FILES:
        p = Path(path)
        if not p.exists():
            print(f"[MISS] {path}")
            continue
        df = pd.read_csv(p)
        try:
            sig = _select_column(df, TORQUE_COLUMN_CANDIDATES)
        except KeyError:
            print(f"[SKIP] {path} (no torque column)")
            continue
        freq, amp = _fft_mag(sig, FS)
        peaks = dominant_peaks(freq, amp, top_k=3)
        plot_path = plot_dir / f"{p.stem}_fft.png"
        try:
            plot_spectrum(freq, amp, str(plot_path), logx=True)
        except Exception as e:
            plot_path = None
            print(f"[WARN] plot failed for {path}: {e}")
        results.append({"file": path, "type": "torque", "peaks": peaks, "plot": str(plot_path) if plot_path else None})
        print(f"[TORQUE] {path} peaks={peaks} plot={plot_path}")

    out = Path("fft_peaks.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[OUT] {out}")


if __name__ == "__main__":
    main()