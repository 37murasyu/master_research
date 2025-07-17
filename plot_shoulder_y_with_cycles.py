from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def unit_scale(unit: str, series: np.ndarray) -> float:
    if unit == 'm':
        return 1.0
    if unit == 'cm':
        return 0.01
    if unit == 'mm':
        return 0.001
    # auto: 推定（中央値の絶対値）
    med = float(np.nanmedian(np.abs(series))) if series.size else 0.0
    if med > 100.0:
        return 0.001
    if med > 1.0:
        return 0.01
    return 1.0


def find_cycle_segments(cycle_index: np.ndarray) -> List[Tuple[int,int,int]]:
    """cycle_index から (start, end, idx) の連続セグメントを抽出（idx>=0のみ）。
    start/end は含む。"""
    segs: List[Tuple[int,int,int]] = []
    if cycle_index is None or len(cycle_index) == 0:
        return segs
    cur_idx = None
    cur_start = None
    for i, v in enumerate(cycle_index):
        if v >= 0:
            if cur_idx is None:
                # 新しいセグメント開始
                cur_idx = int(v)
                cur_start = i
            else:
                # 継続かチェック
                if int(v) != cur_idx:
                    # 変化 -> 直前までがセグメント
                    segs.append((cur_start, i-1, cur_idx))
                    cur_idx = int(v)
                    cur_start = i
        else:
            if cur_idx is not None:
                segs.append((cur_start, i-1, cur_idx))
                cur_idx = None
                cur_start = None
    if cur_idx is not None:
        segs.append((cur_start, len(cycle_index)-1, cur_idx))
    return segs


def main():
    ap = argparse.ArgumentParser(description='肩11のYと検出サイクルを可視化（背景塗りつぶし、検出点マーカー）')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--unit', choices=['auto','m','cm','mm'], default='auto')
    ap.add_argument('--id', type=int, default=11, help='肩のjoint ID（デフォルト 11）')
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--title', type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # X軸: frame列があればそれを使う。なければ行番号
    if 'frame' in df.columns:
        x_full = df['frame'].to_numpy(int)
    else:
        x_full = np.arange(len(df), dtype=int)

    # y列: joint_<id>_y_f が優先、なければ joint_<id>_y
    col_f = f'joint_{args.id}_y_f'
    col = f'joint_{args.id}_y'
    if col_f in df.columns:
        y_full = df[col_f].to_numpy(float)
    elif col in df.columns:
        y_full = df[col].to_numpy(float)
    else:
        raise SystemExit(f'Y列が見つかりません: {col_f} または {col}')

    scale = unit_scale(args.unit, y_full)
    y_full_m = y_full * scale

    # cycle_index 列
    if 'cycle_index' not in df.columns:
        raise SystemExit('cycle_index 列が見つかりません。先に annotate_cycles_in_csv.py を実行してください。')
    c_full = df['cycle_index'].to_numpy(int)

    # ROI
    s = int(args.start) if args.start is not None else 0
    e = int(args.end) if args.end is not None else len(df)
    s = max(0, s); e = max(s+1, min(len(df), e))

    x = x_full[s:e]
    y = y_full_m[s:e]
    c = c_full[s:e]

    # セグメント（ROI内）の抽出
    segs = find_cycle_segments(c)

    # 図作成
    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    # 背景塗り: サイクルごとに色を変える（透明度0.5）
    cmap = plt.get_cmap('tab20')
    for (a, b, idx) in segs:
        # ROI内の座標に合わせて x 値で領域塗り
        xa = x[a]; xb = x[b]
        color = cmap(idx % 20)
        ax.axvspan(xa, xb, color=color, alpha=0.5, linewidth=0)

    # 肩Yのライン
    ax.plot(x, y, color='k', lw=1.5, label=f'joint_{args.id}_y ({args.unit} -> m)')

    # 検出点: 各セグメント開始位置をマーカー
    for (a, b, idx) in segs:
        ax.plot(x[a], y[a], marker='v', color='red', ms=7, label='_nolegend_')

    ax.set_xlabel('frame')
    ax.set_ylabel('shoulder Y (m)')
    ttl = args.title or f'Shoulder {args.id} Y with cycles (alpha=0.5 fill)'
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)

    # 凡例（1つだけ）
    ax.legend(loc='best')

    out_path = args.out or os.path.splitext(args.csv)[0] + f'_shoulder{args.id}_y_cycles.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'[OUT] plot -> {out_path}')


if __name__ == '__main__':
    main()
