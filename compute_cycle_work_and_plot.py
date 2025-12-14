from __future__ import annotations

import argparse
import os
from typing import Dict, List

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
    ap.add_argument('--tau-elbow-npy', required=False, help='右: 肘ローカルトルクy (N,) NPY')
    ap.add_argument('--tau-elbow-npy-left', required=False, help='左: 肘ローカルトルクy (N,) NPY')
    ap.add_argument('--elbow-vec-npy', required=False, help='右: 肘用リンクベクトル (N,3) NPY (省略時は --forearm-npy を使用)')
    ap.add_argument('--elbow-vec-npy-left', required=False, help='左: 肘用リンクベクトル (N,3) NPY (省略時は --forearm-npy-left を使用)')
    ap.add_argument('--offset', type=int, default=6, help='トルクの遅れ補正（フレーム）')
    ap.add_argument('--clip-dtheta', type=float, default=0.35, help='フレーム間角変化のクリップ (rad)')
    ap.add_argument('--body-mass', type=float, required=True, help='体重 M (kg)')
    ap.add_argument('--mmax-all-csv', required=True, help='m_max_all.csv のパス')
    ap.add_argument('--subject-id', type=int, required=True, help='被験者ID（m_max_all.csv の subject_id）')
    ap.add_argument('--out-csv', default='cycle_work.csv', help='サイクル毎の仕事量CSVの出力先')
    ap.add_argument('--out-png', default='cycle_work.png', help='プロットPNGの出力先')
    ap.add_argument('--y-mode', choices=['absolute','percent'], default='percent', help='absolute: [J] で表示, percent: 1RM比(%)で表示')
    return ap.parse_args()


def load_mmax_for_subject(mmax_csv: str, subject_id: int) -> Dict[str, float]:
    df = pd.read_csv(mmax_csv)
    row = df.loc[df['subject_id'] == subject_id]
    if row.empty:
        raise SystemExit(f'subject_id={subject_id} が見つかりません: {mmax_csv}')
    r = row.iloc[0]
    values: Dict[str, float] = {}
    for key in ('wrist_L', 'wrist_R', 'elbow_L', 'elbow_R'):
        if key in r and pd.notna(r[key]):
            try:
                values[key] = float(r[key])
            except (TypeError, ValueError):
                continue
    return values


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

    # 1RM 定数値を部位ごとに算出
    mmax_map = load_mmax_for_subject(args.mmax_all_csv, args.subject_id)
    one_rm_map: Dict[str, float] = {}
    for key, mmax_val in mmax_map.items():
        one_rm_map[key] = compute_1rm_value(args.body_mass, mmax_val)

    rows: List[Dict[str, float]] = []
    series_for_plot: Dict[str, Dict[int, float]] = {}
    one_rm_by_part: Dict[str, float] = {}

    part_configs: List[Dict[str, str]] = []

    if args.forearm_npy and args.tau_npy:
        part_configs.append({
            'part': 'wrist_R',
            'vec_path': args.forearm_npy,
            'tau_path': args.tau_npy,
            'one_rm_key': 'wrist_R',
        })
    if args.forearm_npy_left and args.tau_npy_left:
        part_configs.append({
            'part': 'wrist_L',
            'vec_path': args.forearm_npy_left,
            'tau_path': args.tau_npy_left,
            'one_rm_key': 'wrist_L',
        })

    elbow_vec_R = args.elbow_vec_npy or args.forearm_npy
    if args.tau_elbow_npy and elbow_vec_R:
        part_configs.append({
            'part': 'elbow_R',
            'vec_path': elbow_vec_R,
            'tau_path': args.tau_elbow_npy,
            'one_rm_key': 'elbow_R',
        })
    elbow_vec_L = args.elbow_vec_npy_left or args.forearm_npy_left
    if args.tau_elbow_npy_left and elbow_vec_L:
        part_configs.append({
            'part': 'elbow_L',
            'vec_path': elbow_vec_L,
            'tau_path': args.tau_elbow_npy_left,
            'one_rm_key': 'elbow_L',
        })

    vec_cache: Dict[str, np.ndarray] = {}

    for cfg in part_configs:
        part = cfg['part']
        vec_path = cfg['vec_path']
        tau_path = cfg['tau_path']
        if not vec_path or not tau_path:
            continue
        if not (os.path.exists(vec_path) and os.path.exists(tau_path)):
            missing = []
            if not os.path.exists(vec_path):
                missing.append(vec_path)
            if not os.path.exists(tau_path):
                missing.append(tau_path)
            print(f"[WARN] {part} の入力が不足しています: {', '.join(missing)}")
            continue
        if vec_path not in vec_cache:
            vec_cache[vec_path] = np.load(vec_path)
        vecs = vec_cache[vec_path]
        tau = np.load(tau_path)
        res = compute_wrist_energy(vecs, tau, _dt=1/30.0, offset=args.offset, clip_dtheta=args.clip_dtheta, E_low=0.0, E_high=0.0)
        frames_src = np.arange(len(vecs))
        E_by_cycle = aggregate_cycle_work(frames_src, res.work_inc, cycle_index_map, args.offset)
        series_for_plot[part] = E_by_cycle
        one_rm_key = cfg['one_rm_key']
        one_rm_val = one_rm_map.get(one_rm_key, float('nan'))
        one_rm_by_part[part] = one_rm_val
        for cidx, energy in sorted(E_by_cycle.items()):
            ratio = energy / one_rm_val if one_rm_val and one_rm_val > 0 else np.nan
            rows.append({
                'cycle_index': cidx,
                'part': part,
                'E_cycle_J': energy,
                'one_rm': one_rm_val,
                'ratio_vs_1rm': ratio,
            })

    if not rows:
        raise SystemExit('入力NPYが見つからず、仕事量を計算できませんでした。')

    # CSV 保存
    out_df = pd.DataFrame(rows)
    out_df.sort_values(['part','cycle_index'], inplace=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'[OUT] cycle work CSV -> {args.out_csv}')

    # プロット
    if series_for_plot:
        all_cycles = sorted({c for data in series_for_plot.values() for c in data.keys()})
    else:
        all_cycles = []

    if all_cycles:
        x = np.arange(len(all_cycles))
        parts_for_plot = sorted(series_for_plot.keys())
        width = 0.8 / max(1, len(parts_for_plot))
        offsets = [(idx - (len(parts_for_plot) - 1) / 2) * width for idx in range(len(parts_for_plot))]

        fig, ax = plt.subplots(figsize=(10, 5))
        if args.y_mode == 'percent':
            for idx, part in enumerate(parts_for_plot):
                one_rm_val = one_rm_by_part.get(part, float('nan'))
                denom = one_rm_val if np.isfinite(one_rm_val) and one_rm_val > 0 else np.nan
                if np.isfinite(denom) and denom > 0:
                    y_vals = [(series_for_plot[part].get(c, 0.0) / denom) * 100.0 for c in all_cycles]
                else:
                    y_vals = [0.0 for _ in all_cycles]
                ax.bar(x + offsets[idx], y_vals, width, label=part)
            ax.axhline(100.0, color='gray', linestyle='--', label='1RM=100%')
            ax.set_ylabel('Energy per cycle (% of 1RM)')
            ax.set_title('Work per cycle (percent of 1RM)')
        else:
            max_bar = 0.0
            for idx, part in enumerate(parts_for_plot):
                y_vals = [series_for_plot[part].get(c, 0.0) for c in all_cycles]
                ax.bar(x + offsets[idx], y_vals, width, label=part)
                for val in y_vals:
                    if val > max_bar:
                        max_bar = val
            if max_bar > 0.0:
                ax.set_ylim(0, max_bar * 1.2)
            ax.set_ylabel('Energy per cycle (J)')
            ax.set_title('Work per cycle (J)')

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in all_cycles])
        ax.set_xlabel('cycle_index')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        print(f'[OUT] plot -> {args.out_png}')
    else:
        print('[WARN] 有効なサイクルが見つからなかったため、プロット出力をスキップしました。')


if __name__ == '__main__':
    main()
