from __future__ import annotations

"""
3Dポーズのアニメーション可視化

入力:
  - --csv path/to/pose.csv    (# 列名: joint_<id>_{x,y,z} または joint_<id>_{x,y,z}_f)
  - --npy path/to/pose.npy    (# shape=(T,33,3) を想定)

オプション:
  - --unit {m,cm,mm}          (# 値をmに換算して表示)
  - --stride N                (# フレーム間引き)
  - --max-frames M            (# 上限フレーム)
  - --start S --end E         (# ROI)
  - --save out.mp4|out.gif    (# ファイル保存; 未指定ならインタラクティブ表示)
  - --fps 30                  (# 保存時のFPS)
  - --elev 15 --azim -60      (# 3D視点)

注意:
  - MP4保存にはFFmpegが必要です。インストールされていない場合はGIF保存を推奨します。
"""

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np


def _unit_to_m(unit: str) -> float:
    """Return scale s.t. values_in_unit * _unit_to_m(unit) = meters."""
    if unit == 'm':
        return 1.0
    if unit == 'cm':
        return 0.01
    if unit == 'mm':
        return 0.001
    return 1.0


def load_pose_csv(path: str, joint_ids: Optional[List[int]] = None) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(path)
    if joint_ids is None:
        joint_ids = list(range(33))
    T = len(df)
    pose = np.full((T, 33, 3), np.nan, dtype=float)
    for jid in joint_ids:
        # 優先: *_f -> フォールバック: 素の列
        candidates = [
            (f"joint_{jid}_x_f", f"joint_{jid}_y_f", f"joint_{jid}_z_f"),
            (f"joint_{jid}_x", f"joint_{jid}_y", f"joint_{jid}_z"),
        ]
        cols = None
        for c in candidates:
            if all(col in df.columns for col in c):
                cols = c
                break
        if cols is None:
            continue
        x, y, z = (df[cols[0]].to_numpy(float), df[cols[1]].to_numpy(float), df[cols[2]].to_numpy(float))
        n = min(T, len(x), len(y), len(z))
        pose[:n, jid, 0] = x[:n]
        pose[:n, jid, 1] = y[:n]
        pose[:n, jid, 2] = z[:n]
    return pose


# MediaPipe Pose の主要エッジ（上半身+骨盤〜膝〜足首の一部）。存在しない関節はスキップされます。
EDGES: List[Tuple[int, int]] = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg (上流の一部のみ)
    (24, 26), (26, 28),  # right leg (上流の一部のみ)
]


def animate_pose(
    pose: np.ndarray,
    save_path: Optional[str],
    fps: int,
    elev: float,
    azim: float,
    roll: float = 0.0,
    dpi: int = 120,
    ffmpeg_path: Optional[str] = None,
    show_ids: bool = False,
    id_fontsize: float = 8.0,
    id_color: str = 'crimson',
    id_offset: float = 0.0,
    base_index: int = 0,
    stride_step: int = 1,
    repeat_delay: int = 0,
    vertical_axis: str = 'z',
    axis_triad: bool = False,
    triad_length: Optional[float] = None,
    debug_axes: bool = False,
    axis_unit_label: str = 'm',
):
    import matplotlib
    if save_path:
        matplotlib.use('Agg')  # ヘッドレス保存
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    # ffmpeg のパスを自動/明示指定（必要時）
    # 優先度: 引数 --ffmpeg-path > 環境変数 FFMPEG_PATH > PATH 上の ffmpeg
    try:
        from shutil import which
        detected_ffmpeg = None
        if not ffmpeg_path:
            env_ffmpeg = os.environ.get('FFMPEG_PATH')
            if env_ffmpeg and os.path.exists(env_ffmpeg):
                detected_ffmpeg = env_ffmpeg
        if not ffmpeg_path and not detected_ffmpeg:
            auto = which('ffmpeg')
            if auto:
                detected_ffmpeg = auto
        use_ffmpeg_path = ffmpeg_path or detected_ffmpeg
        if use_ffmpeg_path:
            import matplotlib as mpl
            mpl.rcParams['animation.ffmpeg_path'] = use_ffmpeg_path
    except Exception:
        use_ffmpeg_path = ffmpeg_path or None

    T = pose.shape[0]

    # 軸範囲（自動推定; 余裕を持ってパディング）
    valid = np.isfinite(pose)
    if not valid.any():
        raise SystemExit('No finite pose data to animate')
    mins = np.nanmin(pose, axis=(0, 1))
    maxs = np.nanmax(pose, axis=(0, 1))
    pad = 0.05 * np.max(maxs - mins)
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (mins[2] - pad, maxs[2] + pad)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 初期ビュー: view_init(elev, azim, roll) with fallback for older mpl
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel(f'X ({axis_unit_label})'); ax.set_ylabel(f'Y ({axis_unit_label})'); ax.set_zlabel(f'Z ({axis_unit_label})')
    scat = ax.scatter([], [], [], c='k', s=10)
    title = ax.set_title('')
    # 軸トライアド（原点付近に描画）
    triad_artists = []
    if axis_triad:
        # triad_length が未指定ならデータ範囲の10%を自動採用
        if triad_length is None:
            span = float(np.max(maxs - mins))
            L = 0.1 * span if span > 0 else 0.1
        else:
            L = float(triad_length)
        triad_artists.append(ax.plot([0, L], [0, 0], [0, 0], c='r', lw=2, label='_nolegend_')[0])  # X
        triad_artists.append(ax.plot([0, 0], [0, L], [0, 0], c='g', lw=2, label='_nolegend_')[0])  # Y
        triad_artists.append(ax.plot([0, 0], [0, 0], [0, L], c='b', lw=2, label='_nolegend_')[0])  # Z

    if debug_axes:
        # FFMpegWriter の可用性チェック（参考情報）
        try:
            ffmpeg_available = FFMpegWriter.isAvailable()
        except Exception:
            ffmpeg_available = False
        print(f"[DEBUG] view_init elev={elev} azim={azim} roll={roll} vertical_axis={vertical_axis}")
        print(f"[DEBUG] ffmpeg_path={use_ffmpeg_path if 'use_ffmpeg_path' in locals() else None} available={ffmpeg_available}")

    # 各関節に少なくとも1度でも有限値があるか（時間軸とxyz軸の両方で集約） -> 形状 (33,)
    jid_mask = np.any(np.isfinite(pose), axis=(0, 2))
    present_ids = [j for j in range(pose.shape[1]) if jid_mask[j]]
    # 既定のMediaPipeエッジのうち、少なくとも一方が存在するものに限定
    edges = [(a, b) for (a, b) in EDGES if (a in present_ids and b in present_ids)]
    lines = [ax.plot([], [], [], lw=2, c='tab:blue')[0] for _ in edges]

    # 関節IDテキスト（必要に応じて生成）
    texts = []
    if show_ids:
        for j in range(pose.shape[1]):
            t = ax.text(0.0, 0.0, 0.0, str(j), color=id_color, fontsize=id_fontsize, zorder=5)
            t.set_visible(False)
            texts.append(t)

    def update(i: int):
        P = pose[i]
        pts = np.where(np.isfinite(P[:, 0] + P[:, 1] + P[:, 2]))[0]
        xyz = P[pts]
        if xyz.size:
            scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            scat._offsets3d = ([], [], [])
        # edges
        li = 0
        for (a, b) in edges:
            if not (jid_mask[a] and jid_mask[b]):
                # NaN を設定して確実に非表示化（空配列だと残像が出る場合がある）
                lines[li].set_data([np.nan], [np.nan])
                lines[li].set_3d_properties([np.nan])
            else:
                pa, pb = P[a], P[b]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    lines[li].set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    lines[li].set_3d_properties([pa[2], pb[2]])
                else:
                    lines[li].set_data([np.nan], [np.nan])
                    lines[li].set_3d_properties([np.nan])
            li += 1
        # joint id labels の更新
        if show_ids:
            for j in range(P.shape[0]):
                if not jid_mask[j]:
                    if j < len(texts):
                        texts[j].set_visible(False)
                    continue
                pj = P[j]
                if np.all(np.isfinite(pj)) and j < len(texts):
                    x, y, z = pj
                    dx = id_offset
                    dy = id_offset
                    texts[j].set_position((x + dx, y + dy))
                    try:
                        texts[j].set_3d_properties(z, zdir='z')
                    except Exception:
                        pass
                    texts[j].set_visible(True)
                elif j < len(texts):
                    texts[j].set_visible(False)

        # グローバルフレーム番号（元配列基準, ストライド考慮）
        gidx = base_index + i * max(1, int(stride_step))
        title.set_text(f'global {gidx}  |  segment {i+1}/{T}')
        base_drawn = [scat, *lines, title]
        if show_ids:
            base_drawn.extend(texts)
        if axis_triad:
            base_drawn.extend(triad_artists)
        return base_drawn

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000.0/max(1, fps),
        blit=False,
        repeat=True,
        repeat_delay=max(0, int(repeat_delay)),
    )
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        actual_saved_path = None
        if ext == '.mp4':
            try:
                writer = FFMpegWriter(fps=fps, bitrate=4000)
                ani.save(save_path, writer=writer, dpi=max(50, int(dpi)))
                actual_saved_path = save_path
            except Exception as e:
                print(f'[WARN] mp4保存に失敗: {e}. GIFへフォールバックします。')
                gif_path = os.path.splitext(save_path)[0] + '.gif'
                writer = PillowWriter(fps=fps)
                ani.save(gif_path, writer=writer, dpi=max(50, int(dpi)))
                actual_saved_path = gif_path
        elif ext == '.gif':
            writer = PillowWriter(fps=fps)
            ani.save(save_path, writer=writer, dpi=max(50, int(dpi)))
            actual_saved_path = save_path
        else:
            raise SystemExit('拡張子は .mp4 か .gif を指定してください')
        if actual_saved_path:
            print(f'[OUT] saved -> {actual_saved_path}')
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description='3Dポーズのアニメーション可視化 (CSV/NPY)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--csv', type=str)
    g.add_argument('--npy', type=str)
    # 入力単位（CSV/NPY内の値の単位）と表示単位（グラフラベル/描画の単位）を分離
    ap.add_argument('--in-unit', choices=['m','cm','mm'], default='m', help='入力データの単位（既定: m）')
    ap.add_argument('--out-unit', choices=['m','cm','mm'], default='m', help='表示・保存時の単位（軸ラベルにも反映）')
    # 互換性のためのエイリアス（従来: 値をmに換算して表示、の意味合いだった）
    ap.add_argument('--unit', choices=['m','cm','mm'], default=None, help='[非推奨] 旧オプション。表示単位として扱います。')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--save', type=str, default=None)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--dpi', type=int, default=120, help='保存時のDPI（画質/サイズの調整に使用）')
    ap.add_argument('--elev', type=float, default=15.0)
    ap.add_argument('--azim', type=float, default=-60.0)
    ap.add_argument('--roll', type=float, default=0.0, help='Axes3D.view_init の roll 角（対応バージョンのみ）')
    ap.add_argument('--vertical-axis', type=str, default='z', choices=['x','y','z'], help='グラフラベル上の縦軸表記（デバッグ表示用）')
    ap.add_argument('--show-ids', action='store_true', help='各関節の番号を表示する')
    ap.add_argument('--id-fontsize', type=float, default=8.0)
    ap.add_argument('--id-color', type=str, default='crimson')
    ap.add_argument('--id-offset', type=float, default=0.0, help='テキストをx,y方向へずらす距離（m）')
    ap.add_argument('--repeat-delay', type=int, default=0, help='繰り返し再生時の遅延(ms)')
    ap.add_argument('--axis-triad', action='store_true', help='原点にRGBの軸トライアドを描画 (X=R, Y=G, Z=B)')
    ap.add_argument('--triad-length', type=float, default=0.1, help='軸トライアドの長さ（mスケールに合うように調整）')
    ap.add_argument('--debug-axes', action='store_true', help='view_initや軸情報を標準出力に表示')
    ap.add_argument('--ffmpeg-path', type=str, default=None, help='ffmpeg 実行ファイルのフルパス（MP4保存で必要な場合のみ）')
    args = ap.parse_args()

    # 単位設定
    # --unit が指定された場合は表示単位（out-unit）として扱う（後方互換）
    out_unit = args.out_unit if args.unit is None else args.unit
    in_unit = args.in_unit

    if args.csv:
        # CSVは上半身・下半身のキーが含まれることが多い。最低限 11..16, 23..28 を試す
        pose = load_pose_csv(args.csv, joint_ids=list(range(33)))
    else:
        pose = np.load(args.npy)
        if pose.ndim != 3 or pose.shape[1:] != (33, 3):
            raise SystemExit('NPYは shape=(T,33,3) を想定しています')

    # in-unit → out-unit へ直接変換（例: 入力m, 出力cm -> 1.0 / 0.01 = 100倍）
    s_in = _unit_to_m(in_unit)
    s_out = _unit_to_m(out_unit)
    scale = s_in / max(1e-12, s_out)
    pose = pose.astype(float) * scale

    # ROI & stride & max-frames
    T = pose.shape[0]
    s = max(0, int(args.start))
    e = int(args.end) if args.end is not None else T
    e = max(s+1, min(T, e))
    pose = pose[s:e]
    if args.stride > 1:
        pose = pose[::max(1, args.stride)]
    if args.max_frames is not None and pose.shape[0] > args.max_frames:
        pose = pose[:args.max_frames]

    animate_pose(
        pose,
        args.save,
        args.fps,
        args.elev,
        args.azim,
        roll=args.roll,
        dpi=args.dpi,
        ffmpeg_path=args.ffmpeg_path,
        show_ids=args.show_ids,
        id_fontsize=args.id_fontsize,
        id_color=args.id_color,
        id_offset=args.id_offset,
        base_index=s,
        stride_step=max(1, args.stride),
        repeat_delay=args.repeat_delay,
        vertical_axis=args.vertical_axis,
        axis_triad=args.axis_triad,
        triad_length=args.triad_length,
        debug_axes=args.debug_axes,
        axis_unit_label=out_unit,
    )


if __name__ == '__main__':
    main()
