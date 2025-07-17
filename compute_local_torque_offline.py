"""compute_local_torque_offline.py
=================================================
指定ID(例: 0925_202137) を与えると `output_data/` から
  - kpts3d_<ID>.csv (先頭6行をスキップ, 7行目=フレーム0扱い)
  - aim_torque_vec_<ID>.csv
を自動検出して，フレーム対応させつつグローバルトルクをリンク方向でローカル座標へ変換し
CSV 出力するユーティリティ。

出力: output_data/local_torque_<ID>.csv
列: frame, <part>_x,<part>_y,<part>_z (wrist_R, elbow_R, shoulder_R, wrist_L, elbow_L, shoulder_L)

リンク定義 (master_research_code.py と同じ):
  wrist_R    = joint_4  - joint_2   (手首R - 肘R近位)  -> 前腕方向
  elbow_R    = joint_2  - joint_0   (肘R   - 肩R基準)  -> 上腕方向
  shoulder_R = -(joint_1 - joint_0) (反転させて肩軸)
  wrist_L    = joint_5  - joint_3
  elbow_L    = joint_3  - joint_1
  shoulder_L =  (joint_1 - joint_0)

前提:
- kpts3d CSV の列: frame,joint_0_x,joint_0_y,joint_0_z,...,joint_11_z まで最低 12 関節を含む。
- torque CSV の列: frame, wrist_R_x, wrist_R_y, ... , shoulder_L_z
- フレーム数は kpts3d(有効部) と torque の min に合わせ切り詰め。
- 欠損 (NaN もしくは 非有限) / 極端に短いリンクはローカル変換をスキップし global をそのまま出力。

Usage:
  python compute_local_torque_offline.py 0924_095256
  python compute_local_torque_offline.py 0924_095256 --base-dir output_data --skip-kpts-head 6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from utils import compute_local_torque  # 既存ロジックを流用

# デフォルト (左右分離) スキーマ
DEFAULT_PART_ORDER = [
    "wrist_R",
    "elbow_R",
    "shoulder_R",
    "wrist_L",
    "elbow_L",
    "shoulder_L",
]


def build_torque_cols(part_order: List[str]) -> List[str]:
    return [f"{p}_{ax}" for p in part_order for ax in ("x", "y", "z")]


def _find_file(base_dir: str, prefix: str, id_str: str) -> str | None:
    target = f"{prefix}{id_str}.csv"
    full = os.path.join(base_dir, target)
    return full if os.path.isfile(full) else None


def detect_torque_schema(columns: List[str]) -> Tuple[List[str], List[str]]:
    """列名からスキーマを推定し (part_order, torque_cols) を返す。

    対応スキーマ:
      1. 左右別: wrist_R_x ... shoulder_L_z (6パート = 18列)
      2. 単一側 + 体幹: wrist_x, elbow_x, shoulder_x, core_x, hip_x (5パート = 15列)
    """
    cols = set(columns)
    dual_needed = all(f"{p}_{ax}" in cols for p in DEFAULT_PART_ORDER for ax in ("x", "y", "z"))
    if dual_needed:
        part_order = DEFAULT_PART_ORDER
        return part_order, build_torque_cols(part_order)

    simple_parts = ["wrist", "elbow", "shoulder", "core", "hip"]
    simple_needed = all(f"{p}_{ax}" in cols for p in simple_parts for ax in ("x", "y", "z"))
    if simple_needed:
        return simple_parts, build_torque_cols(simple_parts)

    raise ValueError("Unsupported torque CSV schema: columns=" + ",".join(columns))


def load_torque_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return frames(int), torques(float) and (part_order, torque_cols)."""
    import pandas as pd
    df = pd.read_csv(path)
    part_order, torque_cols = detect_torque_schema(list(df.columns))
    missing = [c for c in (['frame'] + torque_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Torque CSV missing columns: {missing}")
    frames = df['frame'].to_numpy(dtype=np.int64)
    vals = df[torque_cols].to_numpy(dtype=np.float64)
    return frames, vals, part_order, torque_cols


def load_kpts3d_csv(path: str, skip_head: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return frames(int), joints(float) shape (T, J, 3) and joint_ids.

    - MediaPipe のように疎な joint_{ID}_{axis} 列にも対応。
    - フィルタ済みCSVの列名 joint_{ID}_{axis}_f も受理し、存在すればこちらを優先採用。
    - 旧式の連番 (joint_0_*) だけのCSVもサポート。

    skip_head: 先頭無視行数 (6)
    """
    import pandas as pd
    import re
    # 読み込みと先頭スキップ
    df_all = pd.read_csv(path)
    if skip_head > 0:
        df = df_all.iloc[skip_head:].reset_index(drop=True)
    else:
        df = df_all

    if 'frame' not in df.columns:
        raise ValueError("kpts3d CSV に 'frame' 列がありません")

    # joint_{id}_{axis}（および joint_{id}_{axis}_f）を検出（疎インデックス対応、_f を優先）
    joint_ids_set = set()
    for c in df.columns:
        if not c.startswith("joint_"):
            continue
        parts = c.split("_")
        # 期待フォーマット: joint, <id>, <axis>[, f]
        if len(parts) >= 3 and parts[1].isdigit() and parts[2] in ("x", "y", "z"):
            joint_ids_set.add(int(parts[1]))
    joint_ids = sorted(joint_ids_set)

    # 旧式の 0..5 がそろっていない場合でも、最低限 肩/肘/手首(L/R)が揃っていればOK
    if len(joint_ids) == 0:
        raise ValueError("kpts3d CSV に joint_* 列が見つかりません")

    # 必要列が揃っている joint_id のみ使用。_f があれば優先、なければ素の列を使用
    use_ids: List[int] = []
    joint_cols: List[str] = []
    for jid in joint_ids:
        cols_candidates = {
            "x": (f"joint_{jid}_x_f", f"joint_{jid}_x"),
            "y": (f"joint_{jid}_y_f", f"joint_{jid}_y"),
            "z": (f"joint_{jid}_z_f", f"joint_{jid}_z"),
        }
        chosen = []
        ok = True
        for ax in ("x", "y", "z"):
            cand1, cand2 = cols_candidates[ax]
            if cand1 in df.columns:
                chosen.append(cand1)
            elif cand2 in df.columns:
                chosen.append(cand2)
            else:
                ok = False
                break
        if ok:
            use_ids.append(jid)
            joint_cols.extend(chosen)

    if len(use_ids) < 6:
        # 上肢のみCSVでも12関節程度はある想定だが、最小6未満ならエラー
        raise ValueError("Need at least 6 joints in kpts3d CSV (got fewer than 6)")

    frames = df['frame'].to_numpy(dtype=np.int64)
    data = df[joint_cols].to_numpy(dtype=np.float64).reshape(-1, len(use_ids), 3)
    return frames, data, use_ids


def build_links(p3d: np.ndarray, joint_ids: List[int]) -> Dict[str, np.ndarray]:
    """関節配列 p3d と、その各行に対応する joint_ids から、必要なリンクベクトルを構築。

    MediaPipe のIDを優先（R/Lの肩肘手首）：
      R: wrist=16, elbow=14, shoulder=12
      L: wrist=15, elbow=13, shoulder=11
      両肩軸: 12 (右肩), 11 (左肩) -> shoulder_R = (12-11), shoulder_L = (11-12)

    旧式 (0..5) の場合は後方互換で従来定義を使用。
    """
    dual_keys = [
        "wrist_R",
        "elbow_R",
        "shoulder_R",
        "wrist_L",
        "elbow_L",
        "shoulder_L",
    ]
    out = {k: np.array([np.nan, np.nan, np.nan]) for k in dual_keys}
    if p3d.shape[0] == 0:
        return out

    # joint_id -> row index
    id2row = {jid: i for i, jid in enumerate(joint_ids)}

    # 判定: MediaPipe主要IDが揃っていればMediaPipeモード
    has_mp_r = all(j in id2row for j in (16, 14, 12))
    has_mp_l = all(j in id2row for j in (15, 13, 11))
    has_shoulders = all(j in id2row for j in (12, 11))

    if has_mp_r or has_mp_l:
        def g(jid: int) -> np.ndarray:
            return p3d[id2row[jid]] if jid in id2row else np.array([np.nan, np.nan, np.nan])
        if has_mp_r:
            out["wrist_R"] = g(16) - g(14)
            out["elbow_R"] = g(14) - g(12)
        if has_mp_l:
            out["wrist_L"] = g(15) - g(13)
            out["elbow_L"] = g(13) - g(11)
        if has_shoulders:
            # 両肩方向ベクトル
            out["shoulder_R"] = g(12) - g(11)
            out["shoulder_L"] = g(11) - g(12)
        return out

    # 従来(0..5)モード: 必須行が揃っているか確認
    if p3d.shape[0] >= 6 and set(joint_ids[:6]) == set(range(6)):
        out.update({
            "wrist_R": p3d[4] - p3d[2],
            "elbow_R": p3d[2] - p3d[0],
            "shoulder_R": -(p3d[1] - p3d[0]),
            "wrist_L": p3d[5] - p3d[3],
            "elbow_L": p3d[3] - p3d[1],
            "shoulder_L": (p3d[1] - p3d[0]),
        })
    return out


def compute_local_series(kpts: np.ndarray, joint_ids: List[int], torques: np.ndarray, part_order: List[str]) -> np.ndarray:
    """kpts: (T,J,3), joint_ids: (J,), torques: (T,3*P) -> local torques (T,3*P)."""
    T = min(kpts.shape[0], torques.shape[0])
    out = np.zeros((T, torques.shape[1]), dtype=np.float64)
    for t in range(T):
        p3d = kpts[t]
        links_all = build_links(p3d, joint_ids)
        for pi, part in enumerate(part_order):
            gvec = torques[t, pi*3:(pi+1)*3]
            # part 名が wrist/elbow/shoulder (単一側) のときは右側リンクを流用
            if part in ("wrist", "elbow", "shoulder"):
                link_key = part + "_R"
            else:
                link_key = part
            lvec = links_all.get(link_key, np.array([np.nan, np.nan, np.nan]))
            if not np.all(np.isfinite(lvec)) or np.linalg.norm(lvec) < 1e-12:
                out[t, pi*3:(pi+1)*3] = gvec
            else:
                out[t, pi*3:(pi+1)*3] = compute_local_torque(gvec, lvec)
    return out


def save_local_csv(path: str, frames: np.ndarray, local_vals: np.ndarray, torque_cols: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame'] + torque_cols)
        for i in range(local_vals.shape[0]):
            w.writerow([int(frames[i])] + [f"{v:.6f}" for v in local_vals[i]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="kpts3d + torque CSV から局所トルクを再計算し保存")
    ap.add_argument('id', help="例: 0924_095256 (ファイル名末尾) ※ 'kpts3d_<id>.csv' と 'aim_torque_vec_<id>.csv' を探索")
    ap.add_argument('--base-dir', default='output_data', help='CSV を含むディレクトリ')
    ap.add_argument('--skip-kpts-head', type=int, default=6, help='kpts3d の先頭スキップ行数')
    ap.add_argument('--out', default=None, help='出力CSVパス (未指定なら base-dir/local_torque_<id>.csv)')
    args = ap.parse_args(argv)

    kpts_path = _find_file(args.base_dir, 'kpts3d_', args.id)
    torque_path = _find_file(args.base_dir, 'aim_torque_vec_', args.id)
    if not kpts_path or not torque_path:
        print(f"対象ファイルが見つかりません: kpts3d={kpts_path} torque={torque_path}", file=sys.stderr)
        return 1

    try:
        t_frames, t_vals, part_order, torque_cols = load_torque_csv(torque_path)
        k_frames, k_vals, joint_ids = load_kpts3d_csv(kpts_path, args.skip_kpts_head)
    except Exception as e:  # noqa: BLE001
        print(f"読込エラー: {e}", file=sys.stderr)
        return 1

    # フレーム同期: それぞれの frame 列を無視し単純に index 対応 (要望: 7行目=フレーム0)
    T = min(len(t_frames), len(k_frames))
    adj_frames = np.arange(T, dtype=np.int64)

    local_vals = compute_local_series(k_vals[:T], joint_ids, t_vals[:T], part_order)

    out_path = args.out or os.path.join(args.base_dir, f"local_torque_{args.id}.csv")
    save_local_csv(out_path, adj_frames, local_vals, torque_cols)
    print(f"Saved local torque CSV -> {out_path} (frames={T}) parts={part_order}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
