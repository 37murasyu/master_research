from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    root = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE")
    csv_path = root / "output_data" / "noise_stats_summary.csv"
    if not csv_path.exists():
        print(f"[ERR] not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty or "subject" not in df.columns:
        print("[ERR] empty or invalid csv")
        return 1

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("subject")

    subjects = df["subject"].astype(int).to_numpy()

    metrics = [
        ("rho_hf_pose", "HF ratio (pose)"),
        ("rho_hf_torque", "HF ratio (torque)"),
        ("coherence_hf_mean", "HF coherence mean"),
        ("epsilon_p_median", "epsilon_p median"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.ravel()

    for ax, (col, label) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].to_numpy(float)
        ax.plot(subjects, vals, marker="o", linestyle="-", color="#4C78A8")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("subject")

    plt.tight_layout()
    out_png = root / "output_data" / "noise_stats_summary_plot.png"
    plt.savefig(out_png, dpi=150)
    print(f"[OK] {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
