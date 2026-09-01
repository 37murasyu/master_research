from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot noise ratio summary in one figure")
    ap.add_argument("--in-dir", default="output_data/cycle_energy_noise", help="input directory")
    ap.add_argument("--out-png", default="output_data/cycle_energy_noise/noise_ratio_abs_summary.png", help="output png")
    ap.add_argument("--metric", default="noise_ratio_abs", choices=["noise_ratio_abs", "noise_ratio_pos"], help="metric to plot")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("cycle_noise_*.csv"))
    if not files:
        print(f"[ERR] no input files in {in_dir}")
        return 1

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if args.metric not in df.columns:
            continue
        rows.append(df[["cycle_index", "part", args.metric]].copy())

    if not rows:
        print("[ERR] metric not found in inputs")
        return 1

    data = pd.concat(rows, ignore_index=True)
    data = data[np.isfinite(data[args.metric].to_numpy(float))]
    if data.empty:
        print("[ERR] no finite values")
        return 1

    parts = sorted(data["part"].unique())
    values = [data.loc[data["part"] == p, args.metric].to_numpy(float) for p in parts]

    plt.figure(figsize=(10, 5))
    plt.boxplot(values, labels=parts, showfliers=False)
    plt.ylabel(args.metric)
    plt.title(f"Noise ratio summary ({args.metric})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"[OK] {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
