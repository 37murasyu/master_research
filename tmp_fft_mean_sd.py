"""Compute mean±SD FFT spectra across subjects (pose & torque) and plot (log-x, linear y).

- Pose uses 3D magnitude from XYZ columns.
- Torque uses a single column (prefers wrist_R_local_y).
- Uses common length (min across subjects) to align frequency bins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analyze_pushup_fft import compute_fft

FS = 30.0

POSE_FILES: Sequence[str] = (
    "Adjusted 3D Pose/2_stereo_pose.csv",
    "Adjusted 3D Pose/3_0stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/4_0stereo_pose_scaled_with2d.csv",
    "c:/Users/villa/Desktop/master_Research/cameras_raw/5_20250925_133228/5_1stereo_pose_scaled.csv",
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
    "torque/4_0stereo_pose_scaled_with2d_torque.csv",
    "torque/5_1stereo_pose_scaled_torque.csv",
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


def _check_files(paths: Sequence[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))


def _compute_amp_stack(series_list: list[np.ndarray], fs: float) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(s) for s in series_list)
    amps = []
    for s in series_list:
        s_use = np.asarray(s[:min_len], dtype=np.float64)
        freq, amp = compute_fft(s_use, fs)
        amps.append(amp)
    amp_stack = np.vstack(amps)
    return freq, amp_stack


def main() -> None:
    _check_files(POSE_FILES)
    _check_files(TORQUE_FILES)

    # Pose magnitude series
    pose_series = []
    for p in POSE_FILES:
        df = pd.read_csv(p)
        xyz = _select_triplet(df, POSE_TRIPLE_CANDIDATES)
        mag = np.linalg.norm(xyz, axis=1)
        pose_series.append(mag)

    # Torque series
    torque_series = []
    for p in TORQUE_FILES:
        df = pd.read_csv(p)
        sig = _select_column(df, TORQUE_COLUMN_CANDIDATES)
        torque_series.append(sig)

    pose_freq, pose_amp = _compute_amp_stack(pose_series, FS)
    torque_freq, torque_amp = _compute_amp_stack(torque_series, FS)

    # Drop DC for log-x
    pose_freq = pose_freq[1:]
    pose_amp = pose_amp[:, 1:]
    torque_freq = torque_freq[1:]
    torque_amp = torque_amp[:, 1:]

    # geometric mean + percentile band in log-domain
    pose_log = np.log(np.clip(pose_amp, 1e-12, None))
    torque_log = np.log(np.clip(torque_amp, 1e-12, None))

    pose_mean = np.exp(pose_log.mean(axis=0))
    torque_mean = np.exp(torque_log.mean(axis=0))

    pose_p05 = np.exp(np.percentile(pose_log, 5, axis=0))
    pose_p95 = np.exp(np.percentile(pose_log, 95, axis=0))
    torque_p05 = np.exp(np.percentile(torque_log, 5, axis=0))
    torque_p95 = np.exp(np.percentile(torque_log, 95, axis=0))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required for plotting: {exc}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    # Pose plot
    ax = axes[0]
    ax.plot(pose_freq, pose_mean, color="#1f77b4", lw=1.8, label="Pose mean")
    ax.fill_between(pose_freq, pose_p05, pose_p95, color="#1f77b4", alpha=0.2, label="Pose P5–P95")
    ax.set_xscale("log")
    ax.set_xlim(pose_freq[0], FS / 2.0)
    ax.set_ylabel("Amplitude")
    ax.set_title("Pose FFT (mean ± SD)")
    ax.grid(True, alpha=0.3)

    # Torque plot
    ax = axes[1]
    ax.plot(torque_freq, torque_mean, color="#1f77b4", lw=1.8, label="Torque mean")
    ax.fill_between(torque_freq, torque_p05, torque_p95, color="#1f77b4", alpha=0.2, label="Torque P5–P95")
    ax.set_xscale("log")
    ax.set_xlim(torque_freq[0], FS / 2.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Torque FFT (mean ± SD)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = Path("fft_plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mean_sd_pose_torque.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
