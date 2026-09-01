"""Filter pose/torque CSVs with EKF and LPF.

Outputs:
- output_data/filtered_pose_ekf
- output_data/filtered_pose_lpf
- output_data/filtered_torque_ekf
- output_data/filtered_torque_lpf
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from extended_kalman_filter import EKFConfig, run_ekf

FS = 30.0
LPF_FC = 2.0
EKF_CFG = EKFConfig(q_acc=1e-3, r=1e-3, gate_std=3.0)

POSE_DIR = Path("Adjusted 3D Pose")
TORQUE_DIR = Path("torque")
POSE_EXTRA = Path("c:/Users/villa/Desktop/master_Research/cameras_raw/5_20250925_133228/5_1stereo_pose_scaled.csv")

OUT_POSE_EKF = Path("output_data/filtered_pose_ekf")
OUT_POSE_LPF = Path("output_data/filtered_pose_lpf")
OUT_TORQUE_EKF = Path("output_data/filtered_torque_ekf")
OUT_TORQUE_LPF = Path("output_data/filtered_torque_lpf")


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.lower() in ("frame", "time", "timestamp"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _ekf_cols(df: pd.DataFrame, mode: str) -> list[str]:
    cols = _numeric_cols(df)
    if mode == "pose":
        # only position triplets
        return [c for c in cols if c.endswith(('_x', '_y', '_z'))]
    if mode == "torque":
        # limit to main joint components to keep EKF runtime reasonable
        return [c for c in cols if ("wrist" in c.lower() or "elbow" in c.lower()) and c.endswith(('_x', '_y', '_z'))]
    return cols


def _butter_lpf(data: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    if data.size == 0:
        return data
    wn = min(fc / (fs / 2.0), 0.999)
    b, a = butter(order, wn, btype="low")
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = filtfilt(b, a, data[:, i])
    return out


def _ensure_out_dirs() -> None:
    for p in (OUT_POSE_EKF, OUT_POSE_LPF, OUT_TORQUE_EKF, OUT_TORQUE_LPF):
        p.mkdir(parents=True, exist_ok=True)


def _filter_file(path: Path, out_dir_ekf: Path, out_dir_lpf: Path, mode: str) -> None:
    df = pd.read_csv(path)
    cols = _numeric_cols(df)
    if not cols:
        print(f"[SKIP] no numeric cols: {path}")
        return
    df_interp = df.copy()
    df_interp[cols] = df_interp[cols].interpolate(limit_direction="both")
    data = df_interp[cols].to_numpy(float)
    time_s = np.arange(len(df), dtype=float) / FS

    # LPF
    data_lpf = _butter_lpf(data, FS, LPF_FC, order=4)
    df_lpf = df_interp.copy()
    df_lpf[cols] = data_lpf
    out_lpf = out_dir_lpf / f"{path.stem}_lpf.csv"
    df_lpf.to_csv(out_lpf, index=False)

    # EKF (subset to avoid heavy runtime)
    ekf_cols = _ekf_cols(df, mode)
    if ekf_cols:
        data_ekf = df[ekf_cols].to_numpy(float)
        pos, _vel, _acc = run_ekf(data_ekf, time_s, EKF_CFG)
        df_ekf = df.copy()
        df_ekf[ekf_cols] = pos
        out_ekf = out_dir_ekf / f"{path.stem}_ekf.csv"
        df_ekf.to_csv(out_ekf, index=False)
        print(f"[OK] {path.name} -> {out_ekf.name}, {out_lpf.name}")
    else:
        print(f"[OK] {path.name} -> (ekf skipped), {out_lpf.name}")


def _iter_csv(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.glob("*.csv") if p.is_file())


def main() -> None:
    _ensure_out_dirs()

    # Pose
    for p in _iter_csv(POSE_DIR):
        _filter_file(p, OUT_POSE_EKF, OUT_POSE_LPF, mode="pose")
    if POSE_EXTRA.exists():
        _filter_file(POSE_EXTRA, OUT_POSE_EKF, OUT_POSE_LPF, mode="pose")

    # Torque
    for p in _iter_csv(TORQUE_DIR):
        _filter_file(p, OUT_TORQUE_EKF, OUT_TORQUE_LPF, mode="torque")


if __name__ == "__main__":
    main()
