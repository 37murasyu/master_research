r"""plot_pose3d_from_csv.py
CSV (frame, joint_i_x/y/z ...) から 3D ポーズを可視化するユーティリティ。

使い方:
    python plot_pose3d_from_csv.py --csv path\to\stereo_pose.csv --frame 0 --connect

備考:
  - 列名は stereo_triangulate_pose.py が出力する形式を想定
    [frame, joint_0_x, joint_0_y, joint_0_z, ..., joint_32_z]
  - NaN / -1 値はスキップ
  - mediapipe が利用可能なら POSE_CONNECTIONS で骨格線を描画 (--connect)
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np


def load_pose_csv(path: str) -> np.ndarray:
    """CSV を読み (T,J,3) を返す。
    joint_i_x/y/z の存在数から J を推定。
    """
    import pandas as pd

    df = pd.read_csv(path)
    # joint 列探索 (疎なインデックスも許容: joint_16_x など)
    import re
    colnames = list(df.columns)
    # CSV中に現れる順序をなるべく尊重するため、列順に走査して関節IDの出現順を記録
    order_ids: List[int] = []
    id_axes: dict[int, set[str]] = {}
    for name in colnames:
        m = re.fullmatch(r"joint_(\d+)_(x|y|z)", name)
        if not m:
            continue
        jid = int(m.group(1))
        ax = m.group(2)
        if jid not in id_axes:
            id_axes[jid] = set()
            order_ids.append(jid)
        id_axes[jid].add(ax)
    # x,y,z が揃っている関節のみ採用
    use_ids = [jid for jid in order_ids if id_axes.get(jid, set()) == {"x", "y", "z"}]
    if not use_ids:
        raise ValueError("joint_*_* 列が見つかりませんでした")
    # 列の並びは [jid0_x, jid0_y, jid0_z, jid1_x, ...] とする
    joint_cols: List[str] = []
    for jid in use_ids:
        joint_cols += [f"joint_{jid}_x", f"joint_{jid}_y", f"joint_{jid}_z"]
    arr = df[joint_cols].to_numpy(dtype=float)
    T = arr.shape[0]
    J = len(use_ids)
    return arr.reshape(T, J, 3)


def get_mediapipe_connections(J: int) -> List[Tuple[int, int]]:
    """MediaPipe があれば標準の POSE_CONNECTIONS を使用。なければ空。"""
    try:
        import mediapipe as mp  # type: ignore

        # mp.solutions.pose.POSE_CONNECTIONS は関節番号のタプル集合
        conns = list(mp.solutions.pose.POSE_CONNECTIONS)  # type: ignore[attr-defined]
        # 念のため範囲外のものを除外
        return [(a, b) for (a, b) in conns if 0 <= a < J and 0 <= b < J]
    except Exception:
        return []


def plot_frame(points: np.ndarray, connect: bool = True, title: str = "pose3d", save: str | None = None, elev: float = 20.0, azim: float = -60.0):
    """points: (J,3) を 3D 散布図で表示。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3d

    pts = np.asarray(points, dtype=float)
    # 有効点のみ
    valid = np.isfinite(pts).all(axis=1) & (pts != -1).all(axis=1)
    xs, ys, zs = pts[valid, 0], pts[valid, 1], pts[valid, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=np.linspace(0, 1, xs.size), cmap="viridis", s=20)

    if connect:
        conns = get_mediapipe_connections(pts.shape[0])
        for a, b in conns:
            if valid[a] and valid[b]:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], [pts[a, 2], pts[b, 2]], color="gray", linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # 等尺性の調整
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_mid = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_mid = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_mid = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
        ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
        ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=150)
        print(f"Saved: {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="CSVから3Dポーズを描画")
    ap.add_argument("--csv", required=True, help="3DポーズCSVのパス (frame, joint_i_x/y/z...) 規格")
    ap.add_argument("--frame", type=int, default=0, help="表示するフレーム番号")
    ap.add_argument("--no-connect", action="store_true", help="骨格線を描画しない")
    ap.add_argument("--save", type=str, default=None, help="画像として保存するパス (未指定なら画面表示)")
    ap.add_argument("--title", type=str, default=None, help="グラフタイトル")
    ap.add_argument("--elev", type=float, default=20.0, help="視点の仰角")
    ap.add_argument("--azim", type=float, default=-60.0, help="視点の方位角")
    args = ap.parse_args()

    poses = load_pose_csv(args.csv)
    if args.frame < 0 or args.frame >= poses.shape[0]:
        raise IndexError(f"--frame が範囲外です (0..{poses.shape[0]-1})")
    title = args.title or f"{os.path.basename(args.csv)} [frame {args.frame}]"
    plot_frame(poses[args.frame], connect=(not args.no_connect), title=title, save=args.save, elev=args.elev, azim=args.azim)


if __name__ == "__main__":  # pragma: no cover
    main()
