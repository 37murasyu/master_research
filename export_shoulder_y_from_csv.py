from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd

# MediaPipe joint IDs
RIGHT_SHOULDER_ID = 12
LEFT_SHOULDER_ID = 11


def resolve_joint_y_col(df: pd.DataFrame, jid: int) -> str | None:
    candidates = [f"joint_{jid}_y_f", f"joint_{jid}_y"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def unit_scale(unit: str, series: np.ndarray) -> float:
    if unit == 'm':
        return 1.0
    if unit == 'cm':
        return 0.01
    if unit == 'mm':
        return 0.001
    # auto
    med = float(np.nanmedian(np.abs(series))) if series.size else 0.0
    if med > 10.0:
        return 0.001
    if med > 1.0:
        return 0.01
    return 1.0


def main():
    ap = argparse.ArgumentParser(description='CSVから肩(11,12)のY列を抽出し、NPY(メートル換算)で保存')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--unit', choices=['auto','m','cm','mm'], default='auto')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col_r = resolve_joint_y_col(df, RIGHT_SHOULDER_ID)
    col_l = resolve_joint_y_col(df, LEFT_SHOULDER_ID)
    if col_r is None and col_l is None:
        print('[ERR] shoulder Y columns not found')
        sys.exit(1)

    base = os.path.splitext(os.path.basename(args.csv))[0]
    prefix = (args.prefix + '_') if args.prefix else ''

    if col_r is not None:
        yr = df[col_r].to_numpy(float)
        s = unit_scale(args.unit, yr)
        np.save(os.path.join(args.out_dir, f"{prefix}shoulderY_R_{base}.npy"), yr * s)
        print(f"[OUT] R shoulder Y -> {os.path.join(args.out_dir, f'{prefix}shoulderY_R_{base}.npy')}  shape={yr.shape}  scale={s}")
    if col_l is not None:
        yl = df[col_l].to_numpy(float)
        s = unit_scale(args.unit, yl)
        np.save(os.path.join(args.out_dir, f"{prefix}shoulderY_L_{base}.npy"), yl * s)
        print(f"[OUT] L shoulder Y -> {os.path.join(args.out_dir, f'{prefix}shoulderY_L_{base}.npy')}  shape={yl.shape}  scale={s}")


if __name__ == '__main__':
    main()
