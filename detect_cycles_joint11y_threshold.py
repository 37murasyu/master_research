from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Cycle:
    start_valley_idx: int
    peak_idx: int
    end_valley_idx: int


# 既定のしきい値（メートル基準）。オプション未指定時のフォールバック。
THRESH_HIGH = -0.25
THRESH_LOW = -0.40


def _unit_scale(unit: str) -> float:
    if unit == 'm':
        return 1.0
    if unit == 'cm':
        return 0.01
    if unit == 'mm':
        return 0.001
    return 1.0


def find_cycles(y: np.ndarray, thresh_high: float, thresh_low: float) -> List[Cycle]:
    """
    ルール (valley -> peak -> valley):
    - y が thresh_low 以下の極小点を谷とする
    - その後 y が thresh_high 以上の極大点をピークとする
    - さらに次の thresh_low 以下の極小点までを1サイクル
    返り値は start_valley_idx, peak_idx, end_valley_idx のインデックス
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    # 非有限値は線形補間（端は最近傍で埋め）
    mask = np.isfinite(y)
    if not mask.all():
        if mask.any():
            idx = np.arange(T)
            y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
        else:
            return []

    cycles: List[Cycle] = []

    # 単純ピーク検出: 通常の3点比較 + 閾値
    def local_peaks(sig: np.ndarray, min_value: float) -> List[int]:
        peaks = []
        for i in range(1, len(sig) - 1):
            if sig[i] >= sig[i-1] and sig[i] >= sig[i+1] and sig[i] >= min_value:
                peaks.append(i)
        return peaks

    def local_valleys(sig: np.ndarray, max_value: float) -> List[int]:
        valleys = []
        for i in range(1, len(sig) - 1):
            if sig[i] <= sig[i-1] and sig[i] <= sig[i+1] and sig[i] <= max_value:
                valleys.append(i)
        return valleys

    peaks = local_peaks(y, thresh_high)
    valleys = local_valleys(y, thresh_low)

    # valley -> peak -> next valley
    pi = 0
    vi = 0
    while vi < len(valleys):
        v0 = valleys[vi]
        # v0 以降のピークを探す
        while pi < len(peaks) and peaks[pi] <= v0:
            pi += 1
        if pi >= len(peaks):
            break
        p = peaks[pi]
        # p 以降の次の谷 v1 を探す
        vj = vi + 1
        while vj < len(valleys) and valleys[vj] <= p:
            vj += 1
        if vj >= len(valleys):
            break
        v1 = valleys[vj]
        if v0 < p < v1:
            cycles.append(Cycle(start_valley_idx=v0, peak_idx=p, end_valley_idx=v1))
            vi = vj  # 次の探索は v1 以降
            pi += 1
        else:
            vi += 1  # 整合しなければ次の谷へ

    return cycles


def plot_cycles(frames: np.ndarray, y: np.ndarray, cycles: List[Cycle], out_png: Optional[str] = None,
                thr_high: Optional[float] = None, thr_low: Optional[float] = None,
                y_unit_label: str = 'm') -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(frames, y, label=f'joint_11_y ({y_unit_label})', color='tab:blue')
    # 閾値線
    if thr_high is not None:
        plt.axhline(thr_high, color='green', linestyle='--', alpha=0.5, label=f'peak >= {thr_high:.3g} {y_unit_label}')
    if thr_low is not None:
        plt.axhline(thr_low, color='red', linestyle='--', alpha=0.5, label=f'valley <= {thr_low:.3g} {y_unit_label}')

    for c in cycles:
        x0, xp, x1 = frames[c.start_valley_idx], frames[c.peak_idx], frames[c.end_valley_idx]
        y0, yp, y1 = y[c.start_valley_idx], y[c.peak_idx], y[c.end_valley_idx]
        # 区間塗り
        plt.axvspan(x0, x1, color='orange', alpha=0.15)
        # マーカー
        plt.scatter([x0, xp, x1], [y0, yp, y1], c=['red', 'green', 'red'], zorder=5)
        # 境界縦線
        plt.axvline(x0, color='red', alpha=0.5, linestyle=':')
        plt.axvline(xp, color='green', alpha=0.5, linestyle=':')
        plt.axvline(x1, color='red', alpha=0.5, linestyle=':')

    plt.xlabel('frame')
    plt.ylabel(f'joint_11_y ({y_unit_label})')
    plt.legend(loc='best')
    plt.title('joint_11_y cycles (valley->peak->valley)')
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f'[OUT] saved -> {out_png}')
    else:
        plt.show()


def annotate_csv(csv_path: str, cycles: List[Cycle], out_csv: str) -> None:
    """CSVに cycle_index 列を反映。
    out_csv が元CSVと同じ場合は一時ファイルに保存後に置換を試み、
    失敗時は "*_thresholded.csv" にフォールバック。
    """
    import os
    df = pd.read_csv(csv_path)
    T = len(df)
    cycle_index = np.full(T, -1, dtype=int)
    for ci, c in enumerate(cycles, start=1):
        s, e = c.start_valley_idx, c.end_valley_idx
        if 0 <= s < T and 0 <= e < T and s < e:
            cycle_index[s:e+1] = ci
    df['cycle_index'] = cycle_index

    # 同名上書きの安全対応
    same_target = os.path.abspath(out_csv) == os.path.abspath(csv_path)
    if same_target:
        base, ext = os.path.splitext(out_csv)
        tmp_path = base + '.tmp' + ext
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_csv)
            print(f'[OUT] saved -> {out_csv}')
        except Exception as e:
            # フォールバック: 別名で保存
            fb_path = base + '_thresholded' + ext
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            df.to_csv(fb_path, index=False)
            print(f'[WARN] in-place 書き換えに失敗: {e}\n[OUT] fallback -> {fb_path}')
    else:
        df.to_csv(out_csv, index=False)
        print(f'[OUT] saved -> {out_csv}')


def main():
    ap = argparse.ArgumentParser(description='joint_11_y の閾値ピーク・谷・ピークでサイクル検出')
    ap.add_argument('--csv', required=True, help='入力CSV（joint_11_y_f または joint_11_y 列が必要）')
    ap.add_argument('--unit', default='m', choices=['auto','m','cm','mm'], help='CSVの長さ単位（m基準に換算）')
    # しきい値指定（--unit と同じ単位系で解釈）。例: --unit cm --valley-max 42 --peak-min 44
    ap.add_argument('--valley-max', type=float, default=None, help='谷の最大値（以下）で採用するしきい値')
    ap.add_argument('--peak-min', type=float, default=None, help='ピークの最小値（以上）で採用するしきい値')
    ap.add_argument('--out-png', default=None, help='検出結果の可視化PNGパス')
    ap.add_argument('--out-csv', default=None, help='cycle_indexを付与して保存するCSVパス（未指定時は *_with_cycles.csv）')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col = 'joint_11_y_f' if 'joint_11_y_f' in df.columns else 'joint_11_y'
    if col not in df.columns:
        raise SystemExit('CSVに joint_11_y_f も joint_11_y も見つかりません')

    # 単位の自動判定（auto時）: 値の絶対中央値で m/cm/mm の目安を簡易推定
    unit = args.unit
    y_raw = df[col].to_numpy(float)
    if unit == 'auto':
        med = np.nanmedian(np.abs(y_raw))
        if med > 5:      # 5m より大きいなら cm か mm を疑う
            unit = 'cm'
            if med > 50:
                unit = 'mm'
        else:
            unit = 'm'
    scale = _unit_scale(unit)
    y_m = y_raw * scale
    frames = df['frame'].to_numpy(int) if 'frame' in df.columns else np.arange(len(df))

    # サイクル検出
    # しきい値（m単位）を確定
    th_high_m = THRESH_HIGH
    th_low_m = THRESH_LOW
    if args.peak_min is not None:
        th_high_m = float(args.peak_min) * scale
    if args.valley_max is not None:
        th_low_m = float(args.valley_max) * scale
    cycles = find_cycles(y_m, th_high_m, th_low_m)
    print(f'[INFO] detected cycles: {len(cycles)}')

    # プロット
    if args.out_png:
        # プロットは指定単位で表示したいので逆変換する
        inv_scale = 1.0 / max(scale, 1e-12)
        y_disp = y_m * inv_scale
        plot_cycles(frames, y_disp, cycles, args.out_png,
                    thr_high=(th_high_m * inv_scale), thr_low=(th_low_m * inv_scale),
                    y_unit_label=(unit if unit != 'auto' else 'm'))

    # CSV注釈
    out_csv = args.out_csv
    if out_csv is None:
        import os
        base, ext = os.path.splitext(args.csv)
        out_csv = base + '_with_cycles' + ext
    annotate_csv(args.csv, cycles, out_csv)


if __name__ == '__main__':
    main()
