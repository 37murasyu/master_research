"""FFT analysis for push-up pose/angle time series (1D or 3D).

Usage:
    python analyze_pushup_fft.py --input pushup_pose.npy --fs 30 --joint-idx 0 --plot fft.png
    python analyze_pushup_fft.py --input pose.csv --fs 30 --columns elbow_R_angle_smooth_deg

Inputs:
    - NPY: shape (T,), (T,1), (T,3) or (T,J,3) where J is joint count.
    - CSV: 1–3 numeric columns (angles or 3D pos). Pass --columns c1[,c2,c3].
Outputs:
  - Prints dominant frequency and top peaks.
  - Optional amplitude plot saved via --plot.
"""

import argparse
import sys
from typing import List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def _load_npy(path: str, joint_idx: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        # (T, C) where C can be 1 or 3
        if arr.shape[1] in (1, 3):
            return arr
    if arr.ndim == 3 and arr.shape[2] >= 1:
        if joint_idx < 0 or joint_idx >= arr.shape[1]:
            raise ValueError(f"joint_idx {joint_idx} out of range for shape {arr.shape}")
        return arr[:, joint_idx, :]
    raise ValueError(f"Unsupported NPY shape {arr.shape}; expected (T,), (T,1), (T,3) or (T,J,3)")


def _load_csv(path: str, columns: List[str]) -> np.ndarray:
    if pd is None:
        raise ImportError("pandas is required to load CSV. Install pandas or provide NPY input.")
    df = pd.read_csv(path)
    if not 1 <= len(columns) <= 3:
        raise ValueError("columns must list 1 to 3 names")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}; available: {list(df.columns)}")
    return df[columns].to_numpy()


def load_series(path: str, joint_idx: int, columns: List[str]) -> np.ndarray:
    path_lower = path.lower()
    if path_lower.endswith(".npy"):
        data = _load_npy(path, joint_idx)
    elif path_lower.endswith(".csv"):
        data = _load_csv(path, columns)
    else:
        raise ValueError("Unsupported file type. Use .npy or .csv")
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(f"Loaded data has shape {data.shape}; expected (T,1-3)")
    return data


def compute_fft(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.nanmean(sig)
    sig = np.nan_to_num(sig, nan=np.nanmean(sig))
    n = sig.shape[0]
    spec = np.fft.rfft(sig)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(spec) * 2.0 / n
    return freq, amp


def dominant_peaks(freq: np.ndarray, amp: np.ndarray, top_k: int = 3) -> List[Tuple[float, float]]:
    if freq.shape != amp.shape:
        raise ValueError("freq and amp must have the same shape")
    if freq.size <= 1:
        return []
    # Skip DC (freq == 0)
    idx = np.argsort(amp[1:])[::-1][:top_k] + 1
    return [(float(freq[i]), float(amp[i])) for i in idx]


def plot_spectrum(freq: np.ndarray, amp: np.ndarray, path: str, logx: bool = False) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install matplotlib or omit --plot.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq, amp, lw=1.2)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT Spectrum (log-x)" if logx else "FFT Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="FFT analysis for push-up 3D pose time series")
    parser.add_argument("--input", required=True, help="Path to NPY or CSV file")
    parser.add_argument("--fs", type=float, default=30.0, help="Sampling rate [Hz] (default: 30)")
    parser.add_argument("--joint-idx", type=int, default=0, help="Joint index if NPY is (T,J,3)")
    parser.add_argument("--columns", default="x,y,z", help="CSV column names for x,y,z (comma-separated)")
    parser.add_argument("--plot", help="Optional path to save spectrum plot (png)")
    parser.add_argument("--logx", action="store_true", help="Use log scale on frequency axis")
    args = parser.parse_args()

    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    data = load_series(args.input, args.joint_idx, cols)

    # Use vector magnitude as scalar signal (or direct if 1D)
    if data.shape[1] == 1:
        mag = data[:, 0]
    else:
        mag = np.linalg.norm(data[:, :3], axis=1)
    freq, amp = compute_fft(mag, args.fs)

    peaks = dominant_peaks(freq, amp, top_k=5)
    print("Dominant peaks (Hz, amplitude):")
    for f, a in peaks:
        print(f"  {f:.4f} Hz\t{a:.4f}")

    if args.plot:
        plot_spectrum(freq, amp, args.plot, logx=args.logx)
        print(f"Saved plot to {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
