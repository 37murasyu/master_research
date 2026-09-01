from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from detect_cycles_joint11y_threshold import find_cycles, plot_cycles, annotate_csv, _unit_scale


def _pick_shoulder_col(df: pd.DataFrame) -> Tuple[int, str]:
    # MediaPipe: right shoulder joint_12_y, legacy: joint_0_y
    if "joint_12_y" in df.columns:
        return 12, "joint_12_y"
    if "joint_12_y_f" in df.columns:
        return 12, "joint_12_y_f"
    if "joint_0_y" in df.columns:
        return 0, "joint_0_y"
    if "joint_0_y_f" in df.columns:
        return 0, "joint_0_y_f"
    raise ValueError("right shoulder y column not found (joint_12_y or joint_0_y)")


def _auto_thresholds(y: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 0.0
    p10 = float(np.percentile(y, 10))
    p50 = float(np.percentile(y, 50))
    p90 = float(np.percentile(y, 90))
    amp = max(p90 - p10, 1e-6)
    peak_min = p50 + 0.2 * amp
    valley_max = p50 - 0.2 * amp
    return peak_min, valley_max


def _load_series(path: Path, unit: str) -> Tuple[np.ndarray, np.ndarray, int, str]:
    df = pd.read_csv(path)
    jid, col = _pick_shoulder_col(df)
    y_raw = df[col].to_numpy(float)
    frames = df["frame"].to_numpy(int) if "frame" in df.columns else np.arange(len(df))
    if unit == "auto":
        med = np.nanmedian(np.abs(y_raw))
        if med > 5:
            unit = "cm"
            if med > 50:
                unit = "mm"
        else:
            unit = "m"
    scale = _unit_scale(unit)
    y_m = y_raw * scale
    return frames, y_m, jid, unit


def _output_paths(csv_path: Path) -> Tuple[Path, Path]:
    base, ext = os.path.splitext(str(csv_path))
    out_csv = Path(base + "_with_cycles" + ext)
    out_png = Path(base + "_cycles.png")
    return out_csv, out_png


def process_file(path: Path, unit: str) -> None:
    frames, y_m, jid, unit_eff = _load_series(path, unit)
    peak_min, valley_max = _auto_thresholds(y_m)
    cycles = find_cycles(y_m, peak_min, valley_max, order="valley-peak-valley")
    print(f"[INFO] {path.name} cycles={len(cycles)} peak_min={peak_min:.4g} valley_max={valley_max:.4g} ({unit_eff})")

    out_csv, out_png = _output_paths(path)
    # plot in original unit
    inv = 1.0 / max(_unit_scale(unit_eff), 1e-12)
    plot_cycles(
        frames,
        y_m * inv,
        cycles,
        str(out_png),
        thr_high=peak_min * inv,
        thr_low=valley_max * inv,
        y_unit_label=unit_eff,
        joint_id=jid,
    )
    annotate_csv(str(path), cycles, str(out_csv))


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto cycle detection for shoulder y in pose CSVs")
    ap.add_argument("--csv", action="append", required=True, help="pose CSV path (repeatable)")
    ap.add_argument("--unit", default="auto", choices=["auto", "m", "cm", "mm"], help="unit hint")
    args = ap.parse_args()

    for p in args.csv:
        process_file(Path(p), args.unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
