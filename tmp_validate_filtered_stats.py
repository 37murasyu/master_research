import os
from pathlib import Path
import numpy as np
import pandas as pd

FOLDERS = {
    "pose_ekf": Path("output_data/filtered_pose_ekf"),
    "pose_lpf": Path("output_data/filtered_pose_lpf"),
    "torque_ekf": Path("output_data/filtered_torque_ekf"),
    "torque_lpf": Path("output_data/filtered_torque_lpf"),
}


def summarize_file(path: Path):
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return None
    stats = {
        "rows": len(df),
        "cols": num.shape[1],
        "nan_count": int(num.isna().sum().sum()),
        "min": float(np.nanmin(num.values)),
        "p01": float(np.nanpercentile(num.values, 1)),
        "p50": float(np.nanpercentile(num.values, 50)),
        "p99": float(np.nanpercentile(num.values, 99)),
        "max": float(np.nanmax(num.values)),
    }
    return stats


def main():
    for key, folder in FOLDERS.items():
        if not folder.exists():
            print(f"[MISS] {key} {folder}")
            continue
        files = sorted(p for p in folder.glob("*.csv") if p.is_file())
        print(f"[DIR] {key} files={len(files)}")
        for p in files:
            stats = summarize_file(p)
            if stats is None:
                print(f"  [SKIP] {p.name} (no numeric cols)")
                continue
            print(
                f"  {p.name} rows={stats['rows']} cols={stats['cols']} nan={stats['nan_count']} "
                f"min={stats['min']:.3g} p01={stats['p01']:.3g} p50={stats['p50']:.3g} p99={stats['p99']:.3g} max={stats['max']:.3g}"
            )


if __name__ == "__main__":
    main()
