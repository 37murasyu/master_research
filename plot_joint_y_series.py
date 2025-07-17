from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np


def _unit_to_m(unit: str) -> float:
    if unit == 'm':
        return 1.0
    if unit == 'cm':
        return 0.01
    if unit == 'mm':
        return 0.001
    return 1.0


def load_joint_y(csv_path: str, joint_id: int) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    # 優先: *_f -> フォールバック: 素の列
    col_f = f"joint_{joint_id}_y_f"
    col = f"joint_{joint_id}_y"
    if col_f in df.columns:
        y = df[col_f].to_numpy(float)
    elif col in df.columns:
        y = df[col].to_numpy(float)
    else:
        raise SystemExit(f"Y列が見つかりません: {col_f} も {col} も存在しません")
    # 時間軸: frame があれば使う
    if 'frame' in df.columns:
        t = df['frame'].to_numpy(float)
    else:
        t = np.arange(len(y), dtype=float)
    return t, y


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='joint_Y の時系列プロット（PNG保存）')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--joint-id', type=int, required=True)
    ap.add_argument('--unit', choices=['m','cm','mm'], default='m', help='表示単位（軸ラベルのみ）')
    ap.add_argument('--x', choices=['frame','sec'], default='frame', help='横軸をframeか秒にする')
    ap.add_argument('--fps', type=float, default=30.0, help='--x sec のときに使用')
    ap.add_argument('--out-png', required=True)
    args = ap.parse_args(argv)

    t, y = load_joint_y(args.csv, args.joint_id)

    # x 軸
    if args.x == 'sec':
        x = t / max(args.fps, 1e-6)
        xlabel = 'time (s)'
    else:
        x = t
        xlabel = 'frame'

    # y 軸: 表示単位に合わせてスケール（値自体は変換しないでラベルだけでもOKだが、視覚的一貫性のため換算）
    # CSVの単位はm想定
    s_out = _unit_to_m(args.unit)
    scale = 1.0 / max(s_out, 1e-12)
    y_disp = y * scale

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(x, y_disp, lw=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(f'joint_{args.joint_id}_y ({args.unit})')
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f'[OUT] saved -> {args.out_png}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
