import argparse
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd


def _butter_lowpass_filtfilt(x: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    """Low-pass filter with Butterworth+filtfilt. Fallback to moving average if SciPy is unavailable.
    - x: 1D array
    - fs: sampling rate [Hz]
    - fc: cutoff [Hz]
    - order: 2–6 typical
    Returns filtered array (same length). If length too short, returns a copy.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < max(8, 3 * order + 1):
        return x.copy()
    try:
        from scipy.signal import butter, filtfilt  # type: ignore
        nyq = 0.5 * fs
        wn = min(0.99, max(1e-3, fc / nyq))
        b, a = butter(order, wn, btype="low", analog=False)
        try:
            return filtfilt(b, a, x, method="gust")
        except Exception:
            return filtfilt(b, a, x)
    except Exception:
        # Fallback: simple moving average (odd window size)
        k = int(max(3, min(15, round(fs / max(1e-3, fc) * 0.75))))
        if k % 2 == 0:
            k += 1
        k = max(3, min(k, n - (1 - n % 2)))
        w = np.ones(k, dtype=float) / k
        return np.convolve(x, w, mode="same")


_POS_COL_RE = re.compile(r"^joint_\d+_(x|y|z)$")


def _find_position_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if _POS_COL_RE.match(c):
            cols.append(c)
    return cols


def denoise_kpts3d_csv(
    in_csv: str,
    out_csv: Optional[str] = None,
    fps: float = 30.0,
    fc: float = 3.0,
    order: int = 4,
) -> str:
    """Load kpts3d CSV and low-pass filter each joint coordinate (x,y,z).
    - Replaces -1 with NaN, linearly interpolates missing values (both ends), then filters.
    - Writes a new CSV (default: *_filtpos.csv) with the same columns as input, but filtered values.
    Returns output path.
    """
    df = pd.read_csv(in_csv)
    cols = _find_position_columns(df)
    if not cols:
        raise RuntimeError(f"No joint position columns found in {in_csv}")

    out_df = df.copy()
    for c in cols:
        s = pd.to_numeric(out_df[c], errors="coerce").astype(float)
        s = s.mask(s == -1, np.nan)
        # If all NaN, skip
        if not np.isfinite(s).any():
            continue
        # Interpolate NaNs to allow filtering; both-end fill
        s_interp = s.interpolate(limit_direction="both")
        # Some edge cases may still have NaN (e.g., entirely NaN); guard
        arr = s_interp.to_numpy()
        # Replace any residual NaNs with the nearest valid value
        if np.isnan(arr).any():
            # forward fill then back fill as numpy
            mask = np.isnan(arr)
            if (~mask).any():
                idx = np.where(~mask, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]
            else:
                # all NaN, fall back to zeros
                arr = np.zeros_like(arr)

        arr_f = _butter_lowpass_filtfilt(arr, fs=fps, fc=fc, order=order)
        out_df[c] = arr_f

    if out_csv is None:
        base, ext = os.path.splitext(in_csv)
        out_csv = base + "_filtpos" + ext
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def _auto_discover_inputs(root: str) -> List[str]:
    # Look for kpts3d_subject{2..9}_*.csv under root
    paths: List[str] = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not (name.startswith("kpts3d_subject") and name.endswith(".csv")):
                continue
            # skip already-processed or derived files
            if any(sfx in name for sfx in ("_filt.csv", "_filtpos.csv", "_filt_only_f.csv", "_filt_only_f_with_cycles.csv")):
                continue
                # Only subjects 2..9
                try:
                    subj = int(name.split("_subject")[1].split("_")[0])
                except Exception:
                    continue
                if 2 <= subj <= 9:
                    paths.append(os.path.join(root, name))
    return paths


def main():
    ap = argparse.ArgumentParser(description="Denoise 3D pose CSVs (kpts3d) with low-pass filter")
    ap.add_argument("--csv", nargs="*", help="Input CSV paths; if omitted, auto-discover under output_data/poses")
    ap.add_argument("--fps", type=float, default=30.0, help="Sampling rate [Hz] (default: 30)")
    ap.add_argument("--fc", type=float, default=3.0, help="Cutoff frequency [Hz] (default: 3.0)")
    ap.add_argument("--order", type=int, default=4, help="Butterworth order (default: 4)")
    ap.add_argument("--out-suffix", default="_filtpos", help="Output suffix before extension (default: _filtpos)")
    args = ap.parse_args()

    inputs: List[str] = args.csv if args.csv else _auto_discover_inputs(os.path.join("output_data", "poses"))
    if not inputs:
        raise SystemExit("No input CSVs found. Specify --csv or place files under output_data/poses.")

    outs: List[str] = []
    for p in inputs:
        base, ext = os.path.splitext(p)
        out_p = base + args.out_suffix + ext
        out = denoise_kpts3d_csv(p, out_p, fps=args.fps, fc=args.fc, order=args.order)
        print(f"[DONE] {os.path.basename(p)} -> {os.path.basename(out)}")
        outs.append(out)

    print(f"[OUT] written {len(outs)} files")


if __name__ == "__main__":
    main()
