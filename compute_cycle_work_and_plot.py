from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 既存のワーク計算ロジックを再利用
from offline_wrist_energy import compute_wrist_energy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="サイクル毎の仕事量を算出して1RMと併せてプロット")
    ap.add_argument('--cycles-csv', required=True, help='frame と cycle_index を含むCSV（例: *_with_cycles_thresholded.csv）')
    ap.add_argument('--forearm-npy', required=False, help='右: 前腕ベクトル (N,3) NPY')
    ap.add_argument('--tau-npy', required=False, help='右: ローカルトルクy (N,) NPY')
    ap.add_argument('--forearm-npy-left', required=False, help='左: 前腕ベクトル (N,3) NPY')
    ap.add_argument('--tau-npy-left', required=False, help='左: ローカルトルクy (N,) NPY')
    ap.add_argument('--offset', type=int, default=6, help='トルクの遅れ補正（フレーム）')
    ap.add_argument('--clip-dtheta', type=float, default=0.35, help='フレーム間角変化のクリップ (rad)')
    ap.add_argument('--body-mass', type=float, required=True, help='体重 M (kg)')
    ap.add_argument('--mmax-all-csv', required=True, help='m_max_all.csv のパス')
    ap.add_argument('--subject-id', type=int, required=True, help='被験者ID（m_max_all.csv の subject_id）')
    ap.add_argument('--out-csv', default='cycle_work.csv', help='サイクル毎の仕事量CSVの出力先')
    ap.add_argument('--out-png', default='cycle_work.png', help='プロットPNGの出力先')
    ap.add_argument('--y-mode', choices=['absolute','percent'], default='percent', help='absolute: [J] で表示, percent: 1RM比(%)で表示')
    return ap.parse_args()


def load_mmax_for_subject(mmax_csv: str, subject_id: int) -> Tuple[float, float]:
    df = pd.read_csv(mmax_csv)
    row = df.loc[df['subject_id'] == subject_id]
    if row.empty:
        raise SystemExit(f'subject_id={subject_id} が見つかりません: {mmax_csv}')
    r = row.iloc[0]
    return float(r['wrist_L']), float(r['wrist_R'])


def compute_1rm_value(M: float, mmax: float) -> float:
    # 1RM = (6.225e-3 * M + mmax) * 5.01
    return (6.225e-3 * M + mmax) * 5.01


def aggregate_cycle_work(frames: np.ndarray, work_inc: np.ndarray, cycle_index_map: Dict[int, int], offset: int) -> Dict[int, float]:
    """frames: 元フレーム番号 [0..N-1], work_inc は offset適用後の長さ M=N-offset。
    cycle_index_map: frame -> cycle_index。offset 後のフレームのみが対象。
    戻り値: {cycle_index: E_cycle_J}
    """
    E_by_cycle: Dict[int, float] = {}
    for i, w in enumerate(work_inc):
        f = frames[offset + i]
        cidx = cycle_index_map.get(int(f), -1)
        if cidx is None or cidx <= 0:
            continue
        E_by_cycle[cidx] = E_by_cycle.get(cidx, 0.0) + float(w)
    return E_by_cycle


def main():
    args = parse_args()

    # cycles CSV 読込
    df_cyc = pd.read_csv(args.cycles_csv)
    if 'frame' not in df_cyc.columns or 'cycle_index' not in df_cyc.columns:
        raise SystemExit('cycles-csv に frame / cycle_index 列が必要です')
    frames = df_cyc['frame'].to_numpy(int)
    cycle_index_arr = df_cyc['cycle_index'].to_numpy(int)
    cycle_index_map = {int(f): int(c) for f, c in zip(frames, cycle_index_arr)}

    # 1RM 定数値を左右で算出
    mL, mR = load_mmax_for_subject(args.mmax_all_csv, args.subject_id)
    one_rm_L = compute_1rm_value(args.body_mass, mL)
    one_rm_R = compute_1rm_value(args.body_mass, mR)

    rows = []
    series_for_plot = {}

    # 右側
    if args.forearm_npy and args.tau_npy and os.path.exists(args.forearm_npy) and os.path.exists(args.tau_npy):
        fa_R = np.load(args.forearm_npy)
        tauy_R = np.load(args.tau_npy)
        res_R = compute_wrist_energy(fa_R, tauy_R, _dt=1/30.0, offset=args.offset, clip_dtheta=args.clip_dtheta, E_low=0.0, E_high=0.0)
        E_by_cycle_R = aggregate_cycle_work(np.arange(len(fa_R)), res_R.work_inc, cycle_index_map, args.offset)
        for cidx, E in sorted(E_by_cycle_R.items()):
            rows.append({'cycle_index': cidx, 'part': 'wrist_R', 'E_cycle_J': E, 'one_rm': one_rm_R, 'ratio_vs_1rm': (E / one_rm_R) if one_rm_R > 0 else np.nan})
        series_for_plot['R'] = E_by_cycle_R

    # 左側
    if args.forearm_npy_left and args.tau_npy_left and os.path.exists(args.forearm_npy_left) and os.path.exists(args.tau_npy_left):
        fa_L = np.load(args.forearm_npy_left)
        tauy_L = np.load(args.tau_npy_left)
        res_L = compute_wrist_energy(fa_L, tauy_L, _dt=1/30.0, offset=args.offset, clip_dtheta=args.clip_dtheta, E_low=0.0, E_high=0.0)
        E_by_cycle_L = aggregate_cycle_work(np.arange(len(fa_L)), res_L.work_inc, cycle_index_map, args.offset)
        for cidx, E in sorted(E_by_cycle_L.items()):
            rows.append({'cycle_index': cidx, 'part': 'wrist_L', 'E_cycle_J': E, 'one_rm': one_rm_L, 'ratio_vs_1rm': (E / one_rm_L) if one_rm_L > 0 else np.nan})
        series_for_plot['L'] = E_by_cycle_L

    if not rows:
        raise SystemExit('入力NPYが見つからず、仕事量を計算できませんでした。')

    # CSV 保存
    out_df = pd.DataFrame(rows)
    out_df.sort_values(['part','cycle_index'], inplace=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'[OUT] cycle work CSV -> {args.out_csv}')

    # プロット
    # サイクル番号の共通集合
    all_cycles = sorted(set().union(*[set(d.keys()) for d in series_for_plot.values()]))
    x = np.arange(len(all_cycles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    if args.y_mode == 'percent':
        # 1RM比(%)で可視化（左右で1RMが違っても比較しやすい）
        if 'R' in series_for_plot:
            yR = [(series_for_plot['R'].get(c, 0.0) / one_rm_R * 100.0) if one_rm_R > 0 else 0.0 for c in all_cycles]
            ax.bar(x - width/2, yR, width, label='wrist_R')
        if 'L' in series_for_plot:
            yL = [(series_for_plot['L'].get(c, 0.0) / one_rm_L * 100.0) if one_rm_L > 0 else 0.0 for c in all_cycles]
            ax.bar(x + width/2, yL, width, label='wrist_L')
        ax.axhline(100.0, color='gray', linestyle='--', label='1RM=100%')
        ax.set_ylabel('Energy per cycle (% of 1RM)')
        ax.set_title('Wrist work per cycle (percent of 1RM)')
    else:
        # 絶対値[J]表示（バーが小さい場合は見えづらい）。必要に応じてylimを自動スケール
        if 'R' in series_for_plot:
            yR = [series_for_plot['R'].get(c, 0.0) for c in all_cycles]
            ax.bar(x - width/2, yR, width, label='wrist_R')
            ax.axhline(one_rm_R, color='tab:orange', linestyle='--', label=f'1RM_R={one_rm_R:.1f}')
        if 'L' in series_for_plot:
            yL = [series_for_plot['L'].get(c, 0.0) for c in all_cycles]
            ax.bar(x + width/2, yL, width, label='wrist_L')
            ax.axhline(one_rm_L, color='tab:green', linestyle='-.', label=f'1RM_L={one_rm_L:.1f}')
        # バーが極小のときに見えるよう、上限をバー最大の20%上に設定（ただし1RM線ははみ出す可能性あり）
        all_bar = []
        if 'R' in series_for_plot:
            all_bar += yR
        if 'L' in series_for_plot:
            all_bar += yL
        if all_bar:
            ymax = max(all_bar) * 1.2 if max(all_bar) > 0 else 1.0
            ax.set_ylim(0, ymax)
        ax.set_ylabel('Energy per cycle (J)')
        ax.set_title('Wrist work per cycle with 1RM lines')

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_cycles])
    ax.set_xlabel('cycle_index')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f'[OUT] plot -> {args.out_png}')


if __name__ == '__main__':
    main()
