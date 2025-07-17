"""plot_relative_series.py
2つのCSVから同じ列（デフォルト: 2列目）を取り出し、相対的な変動量を折れ線で可視化する。

相対化の方法（--method）:
    - base1: x_t / x_0  → 各系列の最初の値を1に正規化（ご要望）
    - base1-delta: x_t / x_0 - 1 → 基準からの相対変化量
    - mad-diff (既定): |Δx_t| / (MAD(Δx) + eps)
    - pct: |Δx_t| / (|x_{t-1}| + eps)  # パーセント変化に近い
    - zscore-diff: |Δz_t| where z = (x - median) / MAD(x)

使い方例:
  python plot_relative_series.py --csv1 A.csv --csv2 B.csv --col-index 1 --method mad-diff --smooth 3 \
    --save .\\output_data\\rel_change.png

列指定:
  - --col-name を与えれば列名優先
  - なければ --col-index (0基準; 既定は1=2列目)
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np


def read_column(csv_path: str, col_name: Optional[str], col_index: int) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if col_name is not None:
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found in {csv_path}")
        s = df[col_name].to_numpy(dtype=float)
    else:
        if col_index < 0 or col_index >= len(df.columns):
            raise IndexError(f"col-index {col_index} out of range (n_cols={len(df.columns)})")
        s = df.iloc[:, col_index].to_numpy(dtype=float)
    return s


def robust_mad(a: np.ndarray) -> float:
    med = np.nanmedian(a)
    return float(np.nanmedian(np.abs(a - med)))


def rel_change(series: np.ndarray, method: str = "mad-diff", eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    # 基準値（最初の有限値）
    if method in ("base1", "base1-delta"):
        if x.size == 0:
            return x
        # 最初の有限値を見つける
        finite_idx = np.where(np.isfinite(x))[0]
        if finite_idx.size == 0:
            return np.full_like(x, np.nan)
        x0 = x[finite_idx[0]]
        denom = x0 if abs(x0) > eps else eps
        norm = x / denom
        return norm if method == "base1" else (norm - 1.0)

    # 以下は差分ベースの相対化
    diff = np.empty_like(x)
    diff[:] = np.nan
    if x.size >= 2:
        diff[1:] = np.diff(x)

    if method == "mad-diff":
        scale = robust_mad(diff[1:])  # 先頭NaN除く
        scale = scale if scale > 0 else eps
        out = np.abs(diff) / scale
    elif method == "pct":
        base = np.abs(x).copy()
        base[base < eps] = eps
        out = np.abs(diff) / base
    elif method == "zscore-diff":
        mad_x = robust_mad(x)
        mad_x = mad_x if mad_x > 0 else eps
        z = (x - np.nanmedian(x)) / mad_x
        zdiff = np.empty_like(z)
        zdiff[:] = np.nan
        if z.size >= 2:
            zdiff[1:] = np.diff(z)
        out = np.abs(zdiff)
    else:
        raise ValueError(f"unknown method: {method}")
    return out


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    # NaN を無視した移動平均（端は 'same' で処理）
    yy = y.astype(float).copy()
    nan_mask = np.isnan(yy)
    if np.all(nan_mask):
        return yy
    vals = np.where(nan_mask, 0.0, yy)
    valid = (~nan_mask).astype(float)
    kernel = np.ones(int(window), dtype=float)
    num = np.convolve(vals, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    den[den == 0] = 1.0
    smoothed = num / den
    # 元のNaN位置はNaNに戻す
    smoothed[nan_mask] = np.nan
    return smoothed


def plot_two(y1: np.ndarray, y2: np.ndarray, title: str, save: Optional[str], label1: str = "single camera", label2: str = "double cameras"):
    import matplotlib.pyplot as plt
    T = min(len(y1), len(y2))
    x = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(x, y1[:T], label=label1, lw=1.2)
    ax[0].plot(x, y2[:T], label=label2, lw=1.2)
    ax[0].set_ylabel("relative change")
    ax[0].legend()
    # 差の系列（絶対差）
    diff = np.abs(y1[:T] - y2[:T])
    ax[1].plot(x, diff, color="tab:red", lw=1.0)
    ax[1].set_ylabel("|diff|")
    ax[1].set_xlabel("frame")
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Saved: {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="2つのCSVの同列を相対的な変動量で可視化")
    ap.add_argument("--csv1", required=True)
    ap.add_argument("--csv2", required=True)
    ap.add_argument("--col-name", default=None, help="列名（指定時はこちらを優先）")
    ap.add_argument("--col-index", type=int, default=1, help="列番号(0基準)。既定=1=2列目")
    ap.add_argument("--method", choices=["base1", "base1-delta", "mad-diff", "pct", "zscore-diff"], default="mad-diff")
    ap.add_argument("--smooth", type=int, default=1, help="移動平均の窓幅（1=平滑なし）")
    ap.add_argument("--save", default=None, help="画像保存パス（未指定で画面表示）")
    ap.add_argument("--label1", default="single camera", help="凡例ラベル1 (既定: single camera)")
    ap.add_argument("--label2", default="double cameras", help="凡例ラベル2 (既定: double cameras)")
    args = ap.parse_args()

    s1 = read_column(args.csv1, args.col_name, args.col_index)
    s2 = read_column(args.csv2, args.col_name, args.col_index)
    r1 = rel_change(s1, method=args.method)
    r2 = rel_change(s2, method=args.method)
    if args.smooth and args.smooth > 1:
        r1 = smooth_series(r1, args.smooth)
        r2 = smooth_series(r2, args.smooth)
    title = f"relative change ({args.method}) col={'name:'+args.col_name if args.col_name else 'idx:'+str(args.col_index)}"
    plot_two(r1, r2, title, args.save, label1=args.label1, label2=args.label2)


if __name__ == "__main__":  # pragma: no cover
    main()
