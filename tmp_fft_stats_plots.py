"""Create FFT stats plots and mixed-effects table.

Figure A: rho_hf (pose/torque) box+scatter
Figure B: f0 (pose/torque) box+scatter
Table: mixed-effects model (random intercept per subject)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

INPUT_CSV = Path("output_data/noise_stats_summary.csv")
OUT_DIR = Path("fft_plots")
OUT_DIR.mkdir(exist_ok=True)


def _jitter(n: int, scale: float = 0.06) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, scale, size=n)


def _box_scatter(ax, values_pose, values_torque, ylabel: str, title: str):
    data = [values_pose, values_torque]
    ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", alpha=0.2, color="#1f77b4"),
        medianprops=dict(color="#1f77b4", lw=1.8),
        whiskerprops=dict(color="#1f77b4"),
        capprops=dict(color="#1f77b4"),
    )
    ax.scatter(1 + _jitter(len(values_pose)), values_pose, color="#1f77b4", s=28, alpha=0.8)
    ax.scatter(2 + _jitter(len(values_torque)), values_torque, color="#1f77b4", s=28, alpha=0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pose", "Torque"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _mixedlm_table(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df_long = df_long.copy()
    df_long["type"] = pd.Categorical(df_long["type"], categories=["pose", "torque"])
    model = smf.mixedlm(f"{value_col} ~ type", df_long, groups=df_long["subject"])
    # fit without explicit REML to avoid API mismatch warnings across versions
    result = model.fit()
    params = result.params
    bse = result.bse
    pvals = result.pvalues
    out = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "p_value": pvals.values,
        }
    )
    return out


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    df_long_rho = pd.concat(
        [
            df[["subject", "rho_hf_pose"]].rename(columns={"rho_hf_pose": "value"}).assign(type="pose"),
            df[["subject", "rho_hf_torque"]].rename(columns={"rho_hf_torque": "value"}).assign(type="torque"),
        ],
        ignore_index=True,
    )

    df_long_f0 = pd.concat(
        [
            df[["subject", "f0_pose"]].rename(columns={"f0_pose": "value"}).assign(type="pose"),
            df[["subject", "f0_torque"]].rename(columns={"f0_torque": "value"}).assign(type="torque"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    _box_scatter(axes[0], df["rho_hf_pose"].to_numpy(), df["rho_hf_torque"].to_numpy(), r"$\rho_{HF}$", "Figure A: HF ratio")
    _box_scatter(axes[1], df["f0_pose"].to_numpy(), df["f0_torque"].to_numpy(), r"$f_0$ [Hz]", "Figure B: Dominant frequency")

    fig_path = OUT_DIR / "fft_stats_figA_figB.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # Mixed effects tables
    tbl_rho = _mixedlm_table(df_long_rho, "value")
    tbl_f0 = _mixedlm_table(df_long_f0, "value")

    tbl_rho.to_csv(OUT_DIR / "mixedlm_rho_hf.csv", index=False)
    tbl_f0.to_csv(OUT_DIR / "mixedlm_f0.csv", index=False)

    def _to_markdown_table(df_in: pd.DataFrame) -> str:
        headers = [str(c) for c in df_in.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df_in.iterrows():
            lines.append("| " + " | ".join(f"{v}" for v in row.values) + " |")
        return "\n".join(lines)

    md = [
        "# Mixed Effects Model (random intercept per subject)",
        "\n## rho_hf (pose vs torque)\n",
        _to_markdown_table(tbl_rho),
        "\n## f0 (pose vs torque)\n",
        _to_markdown_table(tbl_f0),
        "",
    ]
    (OUT_DIR / "mixedlm_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[OUT] {fig_path}")
    print(f"[OUT] {OUT_DIR / 'mixedlm_rho_hf.csv'}")
    print(f"[OUT] {OUT_DIR / 'mixedlm_f0.csv'}")
    print(f"[OUT] {OUT_DIR / 'mixedlm_summary.md'}")


if __name__ == "__main__":
    main()
