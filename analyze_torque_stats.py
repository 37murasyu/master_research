from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def vec_mag(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    if not all(c in df.columns for c in cols):
        return np.zeros((0,), dtype=float)
    v = df[cols].to_numpy(dtype=float)
    return np.linalg.norm(v, axis=1)


def summarize(m: np.ndarray) -> dict:
    if m.size == 0:
        return {"N": 0}
    return {
        "N": int(m.size),
        "median": float(np.nanmedian(m)),
        "p90": float(np.nanpercentile(m, 90)),
        "p95": float(np.nanpercentile(m, 95)),
        "max": float(np.nanmax(m)),
    }


def main():
    ap = argparse.ArgumentParser(description="Compare torque magnitude stats across CSVs")
    ap.add_argument("--csv", required=True, help="torque CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    parts = ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"]
    out = {}
    for p in parts:
        m = vec_mag(df, p)
        out[p] = summarize(m)

    base = os.path.basename(args.csv)
    print(f"[STATS] {base}")
    for p, s in out.items():
        if s.get("N", 0) == 0:
            continue
        print(f"  {p:12s}  N={s['N']:4d}  median={s['median']:.3f}  p90={s['p90']:.3f}  p95={s['p95']:.3f}  max={s['max']:.3f}")


if __name__ == "__main__":
    main()
