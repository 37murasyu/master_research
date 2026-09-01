import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def remove_outliers_mad(series: pd.Series, thresh: float = 3.5) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad == 0:
        return series  # nothing to clip
    z = 0.6745 * (vals - median) / mad
    keep = np.abs(z) <= thresh
    return pd.Series(vals[keep])


def detect_cycles_threshold(frames: pd.Series, y: pd.Series, upper: float, lower: float):
    """Detect cycles valley->peak->valley with explicit thresholds.

    A cycle starts at a valley (<=lower), reaches a peak (>=upper), and
    ends at the next valley (<=lower). Returns list of (start_frame, end_frame).
    """
    cycles = []
    state = 'seek_valley'  # seek first valley
    valley_start = None
    peak_seen = False
    for f, v in zip(frames, y):
        if state == 'seek_valley':
            if v <= lower:
                valley_start = f
                peak_seen = False
                state = 'seek_peak'
        elif state == 'seek_peak':
            if v >= upper:
                peak_seen = True
                state = 'seek_next_valley'
        elif state == 'seek_next_valley':
            if v <= lower:
                if peak_seen and valley_start is not None:
                    cycles.append((valley_start, f))
                valley_start = f
                peak_seen = False
                state = 'seek_peak'
    return cycles


files = [
    Path('output_data/poses/kpts3d_stereo_24er.csv'),
    Path('output_data/poses/kpts3d_stereo_26eor.csv'),
]

fig, axes = plt.subplots(len(files), 1, figsize=(10, 6), sharex=False)
all_cycles = []  # collect for CSV
annotated = []   # per-file cycle_index data
COL = 'joint_5_y'

for ax, csv_path in zip(axes, files):
    if not csv_path.exists():
        ax.set_title(f"Missing: {csv_path.name}")
        ax.axis('off')
        print(f"[WARN] skip missing {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    if COL not in df.columns:
        ax.set_title(f"{COL} not found: {csv_path.name}")
        ax.axis('off')
        print(f"[WARN] {COL} not found in {csv_path}")
        continue

    y = df[COL]
    frames = df['frame'] if 'frame' in df.columns else pd.Series(range(len(y)))

    # apply frame masks per dataset
    if '26eor' in csv_path.stem:
        mask = (frames >= 600) & (frames <= 1700)
        y = y[mask]
        frames = frames[mask]
    elif '24er' in csv_path.stem:
        mask = frames <= 300
        y = y[mask]
        frames = frames[mask]

    cycles = []
    if '24er' in csv_path.stem:
        y_clean = remove_outliers_mad(y)
        frames_clean = frames.iloc[y_clean.index]
        ax.plot(frames_clean, y_clean, lw=1.0, label=f'{COL} (outliers removed)')
        frames = frames_clean
        y = y_clean
    elif '26eor' in csv_path.stem:
        y_clean = remove_outliers_mad(y)
        frames_clean = frames.iloc[y_clean.index]
        ax.plot(frames_clean, y_clean, lw=1.0, label=f'{COL} (outliers removed)')
        frames = frames_clean
        y = y_clean
        # detect cycles: valley (<= -0.5) -> peak (>= 0.1) -> valley (<= -0.5)
        cycles = detect_cycles_threshold(frames, y, upper=0.1, lower=-0.5)
    else:
        ax.plot(frames, y, lw=1.0, label=COL)

    # add visual spans for cycles
    for start_f, end_f in cycles:
        ax.axvspan(start_f, end_f, color='orange', alpha=0.15)
        all_cycles.append({'file': csv_path.name, 'start_frame': start_f, 'end_frame': end_f})

    # build cycle_index for this CSV (full length of original df)
    cyc_idx = np.full(len(df), -1, dtype=int)
    for idx, (start_f, end_f) in enumerate(cycles):
        in_range = (df['frame'] >= start_f) & (df['frame'] <= end_f)
        cyc_idx[in_range] = idx
    ann_df = df.copy()
    ann_df['cycle_index'] = cyc_idx
    annotated.append((csv_path, ann_df))

    ax.set_title(f"{COL}: {csv_path.name}")
    ax.set_xlabel('frame')
    ax.set_ylabel(COL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

fig.tight_layout()
out_png = Path('output_data/poses/kpts3d_wristY_all.png')
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"[SAVED] {out_png}")

# save cycles to CSV
if all_cycles:
    out_cycles = Path('output_data/poses/kpts3d_wristY_cycles.csv')
    pd.DataFrame(all_cycles).to_csv(out_cycles, index=False)
    print(f"[SAVED] {out_cycles}")

# save annotated CSVs with cycle_index
for orig_path, df_ann in annotated:
    out_path = orig_path.with_name(orig_path.stem + '_with_cycles.csv')
    df_ann.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")
