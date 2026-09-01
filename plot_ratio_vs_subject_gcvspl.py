from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import japanize_matplotlib  # Enable Japanese font rendering

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "output_data" / "cycle_energy"
OUT_PATH = DATA_DIR / "ratio_pos_vs_1rm_by_subject_gcvspl_sd_trim_iqr_shaded_150_180gray.png"

INPUT_FILES = sorted(DATA_DIR.glob("kpts3d_subject*_gcvspl*cycle_work_shoulder.csv"))

rows = []
for path in INPUT_FILES:
    df = pd.read_csv(path)
    if df.empty:
        continue
    if "subject" not in df.columns or "part" not in df.columns:
        continue
    ratio_col = "ratio_pos_vs_1rm" if "ratio_pos_vs_1rm" in df.columns else "ratio_pos_vs_one_rm"
    if ratio_col not in df.columns:
        continue
    df = df.replace([np.inf, -np.inf], np.nan)
    for part in ("elbow_R", "wrist_R"):
        sub = df[df["part"] == part][ratio_col].dropna().to_numpy(float)
        if sub.size == 0:
            continue
        rows.append({"subject": int(df["subject"].iloc[0]), "part": part, "ratio": sub})

if not rows:
    raise SystemExit("[WARN] no data to plot")

# aggregate per subject and part
subj = {}
for r in rows:
    key = (r["subject"], r["part"])
    subj.setdefault(key, []).append(r["ratio"])

subjects = sorted({k[0] for k in subj.keys()})
parts = ["elbow_R", "wrist_R"]

data_by_part = {p: [] for p in parts}
for s in subjects:
    for p in parts:
        arrs = subj.get((s, p), [])
        vals = np.concatenate(arrs) if arrs else np.array([], dtype=float)
        data_by_part[p].append(vals)


def _trim_outliers_iqr(vals: np.ndarray, k: float = 1.5) -> np.ndarray:
    if vals.size == 0:
        return vals
    q1, q3 = np.nanpercentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return vals[(vals >= lo) & (vals <= hi)]


def _stats_from_vals(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {
            "med": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "whislo": np.nan,
            "whishi": np.nan,
        }
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    q1 = max(mean - std, vmin)
    q3 = min(mean + std, vmax)
    return {
        "med": mean,
        "q1": q1,
        "q3": q3,
        "whislo": vmin,
        "whishi": vmax,
    }


trimmed_by_part = {p: [] for p in parts}
for p in parts:
    for vals in data_by_part[p]:
        trimmed_by_part[p].append(_trim_outliers_iqr(vals))

stats_elbow = [_stats_from_vals(v) for v in trimmed_by_part["elbow_R"]]
stats_wrist = [_stats_from_vals(v) for v in trimmed_by_part["wrist_R"]]

fig, ax = plt.subplots(figsize=(11, 5))
positions = np.arange(len(subjects))
offset = 0.18


def _draw_boxes(stats_list: list[dict], x_positions: np.ndarray, color: str) -> None:
    width = 0.3
    cap = width * 0.5
    for x, st in zip(x_positions, stats_list):
        if np.isnan(st["q1"]) or np.isnan(st["q3"]) or np.isnan(st["med"]):
            continue
        rect = Rectangle(
            (x - width / 2, st["q1"]),
            width,
            st["q3"] - st["q1"],
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.plot([x - width / 2, x + width / 2], [st["med"], st["med"]], color="black", linewidth=1.5)
        ax.plot([x, x], [st["whislo"], st["q1"]], color="black", linewidth=1.2)
        ax.plot([x, x], [st["q3"], st["whishi"]], color="black", linewidth=1.2)
        ax.plot([x - cap / 2, x + cap / 2], [st["whislo"], st["whislo"]], color="black", linewidth=1.2)
        ax.plot([x - cap / 2, x + cap / 2], [st["whishi"], st["whishi"]], color="black", linewidth=1.2)


_draw_boxes(stats_elbow, positions - offset, "#4C78A8")
_draw_boxes(stats_wrist, positions + offset, "#F58518")

ax.axhspan(0, 1.5, facecolor="#F6C89F", alpha=0.18, zorder=0)  # shade up to 150%
ax.axhspan(1.5, 1.8, facecolor="#B0B0B0", alpha=0.15, zorder=0)  # shade 150% to 180%
ax.axhline(1.0, color="#808080", linestyle="--", linewidth=2)
ax.set_xticks(positions)
ax.set_xticklabels([str(s) for s in subjects])
ax.set_xlabel("被験者")
ax.set_ylabel("比率 (%)")
ax.set_title("正の仕事 / 1RM（SD、外れ値トリム後、%）")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.legend(
    handles=[
        Patch(facecolor="#4C78A8", edgecolor="black", label="肘（右）"),
        Patch(facecolor="#F58518", edgecolor="black", label="手首（右）"),
    ],
    loc="upper right",
)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()

DATA_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
print(f"[OK] {OUT_PATH}")
