r"""filter_pose3d.py
3D姿勢CSVに対して、欠損補間 → 平滑（ゼロ位相） → 微分（速度・加速度、上腕/前腕の角速度・角加速度）を行い、別ファイルで保存。

機能:
- 短欠損補間: ~0.2s 以下を線形補間（--gap-max-sec）。それより長い欠損はスプライン/近傍回帰で補間（SciPyが無い場合は線形のまま、もしくはNaN維持）。
- 平滑（ゼロ位相）:
  - Butterworth 4次 + filtfilt（優先、SciPyが無い場合は自動でSavitzky–Golayへフォールバック）
  - 代替: Savitzky–Golay（対称窓・次数指定）
  - 代替: One Euro Filter（min_cutoff, beta, dcutoff）
- 微分: 平滑後に中心差分で速度/加速度。端点はNaN、valid_diff列で有効区間を明示。
- 角速度/角加速度（上腕/前腕）: u×du/dt, u×d²u/dt² を出力（MediaPipe ID: R=12-14-16, L=11-13-15）。

出力:
- 入力 foo_pose.csv → foo_pose_filt.csv（既定）

例:
  python filter_pose3d.py \
    --csv .\output_data\poses\stereo_9_20250925_201442_pose.csv \
    --fps 30 --method butter --butter-order 4 \
    --gap-max-sec 0.2 --save .\output_data\poses\stereo_9_20250925_201442_pose_filt.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


# ===== Utilities =====
def try_import_scipy():
    try:
        import scipy.signal as sig  # type: ignore
        import scipy.interpolate as sinterp  # type: ignore
        return sig, sinterp
    except (ImportError, ModuleNotFoundError):
        return None, None


def central_diff_series(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(x)
    dx = np.full(T, np.nan, dtype=float)
    ddx = np.full(T, np.nan, dtype=float)
    if T >= 3:
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        ddx[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt * dt)
        # 端点は一次/二次の片側差分。必要ならNaNのままでもよいが一応埋める。
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
        ddx[0] = ddx[1]
        ddx[-1] = ddx[-2]
    elif T == 2:
        dx[:] = (x[1] - x[0]) / dt
        ddx[:] = np.nan
    else:
        pass
    return dx, ddx


def unit_vec(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def interpolate_short_gaps(y: np.ndarray, max_len: int) -> np.ndarray:
    """線形補間で短欠損(<=max_len)を埋める。長欠損はNaNのまま。
    y: (T,) with NaNs
    """
    yy = y.copy()
    T = len(yy)
    isn = np.isnan(yy)
    if not isn.any():
        return yy
    # 連続NaN区間を走査
    i = 0
    while i < T:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < T and isn[j]:
            j += 1
        gap = j - i
        if gap <= max_len:
            # 端の外挿は避ける（両側に有効値が必要）
            left = i - 1
            right = j
            if left >= 0 and right < T and not np.isnan(yy[left]) and not np.isnan(yy[right]):
                yy[i:j] = np.linspace(yy[left], yy[right], gap + 2)[1:-1]
        # 次区間へ
        i = j
    return yy


def interpolate_long_gaps(y: np.ndarray, window: int = 30) -> np.ndarray:
    """長欠損を近傍点からスプライン/回帰で補間（SciPyが無ければそのまま）。
    window: 近傍片側点数
    """
    yy = y.copy()
    T = len(yy)
    isn = np.isnan(yy)
    if not isn.any():
        return yy
    _sig, sinterp = try_import_scipy()
    if sinterp is None:
        # フォールバック: 何もしない（長欠損は解析から除外）
        return yy
    x = np.arange(T)
    i = 0
    while i < T:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < T and isn[j]:
            j += 1
        left = max(0, i - window)
        right = min(T, j + window)
        xs = x[left:i].tolist() + x[j:right].tolist()
        ys = yy[left:i].tolist() + yy[j:right].tolist()
        xs = np.array(xs)
        ys = np.array(ys)
        mask = np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) >= 5:
            try:
                spl = sinterp.UnivariateSpline(xs, ys, s=0.0, k=min(3, len(xs) - 1))
                yy[i:j] = spl(x[i:j])
            except (ValueError, RuntimeError):
                # 補間失敗時はその区間は未補間のままにする
                pass
        i = j
    return yy


def apply_smoothing(series: np.ndarray, method: str, fps: float, fc: float,
                    butter_order: int = 4, sg_window: int = 15, sg_poly: int = 3,
                    one_euro_params: Tuple[float, float, float] = (1.2, 0.1, 1.0)) -> np.ndarray:
    y = series.copy()
    if method == 'butter':
        sig, _ = try_import_scipy()
        if sig is None:
            method = 'sg'  # フォールバック
        else:
            wn = fc / (0.5 * fps)
            wn = min(max(wn, 1e-4), 0.99)
            b, a = sig.butter(butter_order, wn, btype='low', analog=False)
            # NaNハンドリング: 一時補間してからfiltfilt、後でNaN位置に戻す
            nanmask = ~np.isfinite(y)
            if nanmask.any():
                y = np.interp(np.arange(len(y)), np.flatnonzero(~nanmask), y[~nanmask])
            y = sig.filtfilt(b, a, y, method='gust')
            y[nanmask] = np.nan
            return y
    if method == 'sg':
        sig, _ = try_import_scipy()
        if sig is None:
            # 単純移動平均（ゼロ位相ではないが簡易代替）
            k = max(3, sg_window | 1)
            yy = y.copy()
            nanmask = ~np.isfinite(yy)
            if nanmask.any():
                yy = np.interp(np.arange(len(yy)), np.flatnonzero(~nanmask), yy[~nanmask])
            kernel = np.ones(k) / k
            yy = np.convolve(yy, kernel, mode='same')
            yy[nanmask] = np.nan
            return yy
        else:
            k = max(3, sg_window)
            if k % 2 == 0:
                k += 1
            nanmask = ~np.isfinite(y)
            if nanmask.any():
                y = np.interp(np.arange(len(y)), np.flatnonzero(~nanmask), y[~nanmask])
            yy = sig.savgol_filter(y, window_length=k, polyorder=sg_poly, mode='interp')
            yy[nanmask] = np.nan
            return yy
    if method == 'oneeuro':
        min_cutoff, beta, dcutoff = one_euro_params
        # 実装: https://cristal.univ-lille.fr/~casiez/1euro/
        def alpha(cutoff: float) -> float:
            tau = 1.0 / (2 * np.pi * cutoff)
            te = 1.0 / fps
            return 1.0 / (1.0 + tau / te)
        a_d = alpha(dcutoff)
        dx = np.zeros_like(y)
        for t in range(1, len(y)):
            dx[t] = a_d * (y[t] - y[t-1]) + (1 - a_d) * dx[t-1]
        out = np.zeros_like(y)
        prev = y[0] if np.isfinite(y[0]) else 0.0
        for t in range(len(y)):
            cutoff = min_cutoff + beta * abs(dx[t])
            a = alpha(cutoff)
            cur = y[t] if np.isfinite(y[t]) else prev
            prev = a * cur + (1 - a) * prev
            out[t] = prev
        out[~np.isfinite(series)] = np.nan
        return out
    return y


def joint_fc_map(jid: int, default_fc: float, wrists_fc: float, elbows_fc: float,
                 shoulders_fc: float, pelvis_fc: float, knees_fc: float, ankles_fc: float) -> float:
    if jid in (23, 24):  # pelvis/hip baseline (左右)
        return pelvis_fc
    if jid in (11, 12):  # shoulders
        return shoulders_fc
    if jid in (13, 14):  # elbows
        return elbows_fc
    if jid in (15, 16):  # wrists
        return wrists_fc
    if jid in (25, 26):  # knees
        return knees_fc
    if jid in (27, 28):  # ankles
        return ankles_fc
    return default_fc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3D姿勢CSVの補間・平滑・微分を実行して保存")
    ap.add_argument('--csv', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--gap-max-sec', type=float, default=0.2, help='線形補間する欠損の最大長[s]')
    ap.add_argument('--method', choices=['butter', 'sg', 'oneeuro'], default='butter')
    ap.add_argument('--butter-order', type=int, default=4)
    ap.add_argument('--sg-window', type=int, default=15)
    ap.add_argument('--sg-poly', type=int, default=3)
    ap.add_argument('--oneeuro-min-cutoff', type=float, default=1.2)
    ap.add_argument('--oneeuro-beta', type=float, default=0.1)
    ap.add_argument('--oneeuro-dcutoff', type=float, default=1.0)
    # カットオフ（Hz）推奨例を既定値として与える
    ap.add_argument('--fc-default', type=float, default=3.0)
    ap.add_argument('--fc-wrists', type=float, default=2.5)
    ap.add_argument('--fc-elbows', type=float, default=3.0)
    ap.add_argument('--fc-shoulders', type=float, default=3.0)
    ap.add_argument('--fc-pelvis', type=float, default=4.0)
    ap.add_argument('--fc-knees', type=float, default=3.0)
    ap.add_argument('--fc-ankles', type=float, default=2.5)
    ap.add_argument('--save', default=None)
    ap.add_argument('--only-f', action='store_true', help='frame と joint_*_f のみ保存（軽量出力）')
    args = ap.parse_args(argv)

    import pandas as pd
    df = pd.read_csv(args.csv)
    if 'frame' not in df.columns:
        raise ValueError("CSVは 'frame' 列を含む必要があります")

    # joint_{id}_{axis} を収集
    joint_ids = sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('joint_') and c.split('_')[-1] in ('x','y','z') and c.split('_')[1].isdigit()})
    if not joint_ids:
        raise ValueError('joint_* 列が見つかりません')

    frames = df['frame'].to_numpy(dtype=np.int64)
    T = len(frames)
    dt = 1.0 / max(args.fps, 1e-6)
    # 欠損補間の最大長（サンプル数）
    max_gap = int(round(args.gap_max_sec * args.fps))

    # 出力データフレームの準備
    out = {'frame': frames}
    valid_diff = np.zeros(T, dtype=int)
    if T >= 3:
        valid_diff[1:-1] = 1
    out['valid_diff'] = valid_diff

    # 平滑パラメータ
    oneeuro_params = (args.oneeuro_min_cutoff, args.oneeuro_beta, args.oneeuro_dcutoff)

    # 各ジョイント・各軸で処理
    for jid in joint_ids:
        fc = joint_fc_map(jid, args.fc_default, args.fc_wrists, args.fc_elbows, args.fc_shoulders, args.fc_pelvis, args.fc_knees, args.fc_ankles)
        for ax in ('x','y','z'):
            col = f'joint_{jid}_{ax}'
            y = df[col].to_numpy(dtype=float)
            # 欠損補間
            y1 = interpolate_short_gaps(y, max_gap)
            y2 = interpolate_long_gaps(y1, window=int(0.5*args.fps))
            # 平滑
            y3 = apply_smoothing(y2, args.method, args.fps, fc, args.butter_order, args.sg_window, args.sg_poly, oneeuro_params)
            # 微分
            vy, ay = central_diff_series(y3, dt)
            out[f'{col}_f'] = y3
            out[f'{col}_v'] = vy
            out[f'{col}_a'] = ay

    # 角速度/角加速度（上腕/前腕）
    def coln(j, a):
        return f'joint_{j}_{a}_f'
    def get_vec(j):
        return np.column_stack([out.get(coln(j,'x')), out.get(coln(j,'y')), out.get(coln(j,'z'))])

    def ang_deriv(P, Q):  # 単位ベクトル u=unit(Q-P) の角速度/角加速度ベクトル
        u = unit_vec(Q - P)
        # 各軸の時間微分をまとめて計算（未代入の可能性を排除）
        du_mat = np.zeros((T, 3), dtype=float)
        dd_mat = np.zeros((T, 3), dtype=float)
        for k in range(3):
            du, ddu = central_diff_series(u[:, k], dt)
            du_mat[:, k] = du
            dd_mat[:, k] = ddu
        # ω = u × du/dt, α = u × d²u/dt²（軸方向）
        wv = np.cross(u, du_mat)
        av = np.cross(u, dd_mat)
        return wv, av

    # R: 12-14-16, L: 11-13-15
    have = set(joint_ids)
    sides = []
    if {12,14,16}.issubset(have):
        sides.append(('R', 12, 14, 16))
    if {11,13,15}.issubset(have):
        sides.append(('L', 11, 13, 15))

    for tag, sh, el, wr in sides:
        S = get_vec(sh)
        E = get_vec(el)
        W = get_vec(wr)
        if S is None or E is None or W is None:
            continue
        w_upper, a_upper = ang_deriv(S, E)
        w_fore, a_fore = ang_deriv(E, W)
        for comp, name in [(w_upper,'upper'), (a_upper,'upper_a'), (w_fore,'fore'), (a_fore,'fore_a')]:
            out[f'{tag}_'+name+'_wx'] = comp[:,0]
            out[f'{tag}_'+name+'_wy'] = comp[:,1]
            out[f'{tag}_'+name+'_wz'] = comp[:,2]

    # 保存
    # 出力カラム選択
    if args.only_f:
        # frame と *_f のみ
        cols = ['frame'] + [c for c in out.keys() if c.endswith('_f') and c.startswith('joint_')]
        out_df = pd.DataFrame({c: out[c] for c in cols})
    else:
        out_df = pd.DataFrame(out)
    save_path = args.save
    if not save_path:
        base, _ = os.path.splitext(args.csv)
        save_path = f"{base}_filt.csv"
        if args.only_f:
            save_path = f"{base}_filt_only_f.csv"
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    out_df.to_csv(save_path, index=False)
    print(f"Saved filtered pose CSV: {save_path} (T={T}, joints={len(joint_ids)})")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
