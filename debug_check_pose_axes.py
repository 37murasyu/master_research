from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def stats(name: str, a: np.ndarray) -> str:
    return f"{name}: shape={a.shape} min={np.nanmin(a):.4f} max={np.nanmax(a):.4f} mean={np.nanmean(a):.4f} std={np.nanstd(a):.4f}"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float('nan')
    aa = a[m] - a[m].mean()
    bb = b[m] - b[m].mean()
    denom = np.sqrt((aa**2).sum() * (bb**2).sum())
    if denom == 0:
        return float('nan')
    return float((aa * bb).sum() / denom)


def main():
    ap = argparse.ArgumentParser(description="CSV内の joint_11/12 の x_f と y_f の関係を検査（等しい/相関）")
    ap.add_argument('--csv', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    T = len(df)
    s = max(0, int(args.start))
    e = int(args.end) if args.end is not None else T
    e = max(s+1, min(T, e))
    df = df.iloc[s:e]

    for jid in (11, 12):
        cx = f"joint_{jid}_x_f" if f"joint_{jid}_x_f" in df.columns else f"joint_{jid}_x"
        cy = f"joint_{jid}_y_f" if f"joint_{jid}_y_f" in df.columns else f"joint_{jid}_y"
        if cx not in df.columns or cy not in df.columns:
            print(f"[WARN] joint {jid}: columns not found")
            continue
        x = df[cx].to_numpy(float)
        y = df[cy].to_numpy(float)
        print(f"[JOINT {jid}] {stats('x', x)} | {stats('y', y)} | corr(x,y)={corr(x,y):.4f}")
        eq = np.allclose(x, y, rtol=0, atol=1e-9)
        print(f"  allclose(x,y): {eq}")


if __name__ == '__main__':
    main()
