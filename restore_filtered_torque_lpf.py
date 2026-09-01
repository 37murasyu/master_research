"""Restore LPF torque CSVs from source torque CSVs for selected files."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS = 30.0
LPF_FC = 2.0

OUT_DIR = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_torque_lpf")
SRC_DIR = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\torque")

FILES = [
    "2_stereo_pose_torque.csv",
    "3_0stereo_pose_scaled_with2d_torque.csv",
    "3_1stereo_pose_scaled_with2d_torque.csv",
    "4_0stereo_pose_scaled_with2d_torque.csv",
]


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.lower() in ("frame", "time", "timestamp"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = SRC_DIR / name
        df = pd.read_csv(src)
        cols = _numeric_cols(df)
        if not cols:
            print(f"[SKIP] no numeric cols: {src}")
            continue
        df_interp = df.copy()
        df_interp[cols] = df_interp[cols].interpolate(limit_direction="both")
        data_lpf = _butter_lpf(df_interp[cols].to_numpy(float), FS, LPF_FC, order=4)
        df_lpf = df_interp.copy()
        df_lpf[cols] = data_lpf
        out = OUT_DIR / f"{src.stem}_lpf.csv"
        df_lpf.to_csv(out, index=False)
        print(f"[OK] {src.name} -> {out}")


if __name__ == "__main__":
    main()
