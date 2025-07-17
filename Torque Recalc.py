# %%
import os
import glob
import numpy as np
import pandas as pd

from config import dt, w, m1, m2, m4, g, folder_path
from link_vector_calculator_module import LinkVectorCalculator
from utils_dynamic import (
    calculate_inertia_tensor,
    calculate_M_and_F,
    calculate_individual_torques,
)

# ローカル変換関数をインポート（モジュール名を適宜修正）
from utils import compute_local_torque  # TODO: 調整
from body_part_storage_module import BodyPartDataStorage

out_dir = os.path.join(folder_path, "output_data")
# --- 設定 ---
# 姿勢データ CSV と トルクデータ CSV は別ファイル
pose_csv_path = (glob.glob(os.path.join(out_dir, "kpts3d_0427_204642_nakamura22.csv")))[
    0
]
torque_csv_path = (
    glob.glob(os.path.join(out_dir, "aim_torque_vec_0427_204642_nakamura22.csv"))
)[0]


# --- CSV読み込み ---
pos_df = pd.read_csv(pose_csv_path)
torque_df = pd.read_csv(torque_csv_path)
# --- フレーム数差によるオフセット自動判定 ---
n_pos = pos_df["frame"].nunique()
n_tor = torque_df["frame"].nunique()
diff = n_pos - n_tor

if diff == 6:
    offset = 6
elif diff == 0:
    offset = 0
else:
    raise ValueError(f"Frame count mismatch: pose={n_pos}, torque={n_tor}")
# --- joint0z 列と極小値フラグを姿勢データから計算 ---
pos_df["joint0z"] = pos_df["joint_0_z"]
pos_df["is_local_min_joint0_z"] = (pos_df["joint0z"] < pos_df["joint0z"].shift(1)) & (
    pos_df["joint0z"] < pos_df["joint0z"].shift(-1)
)
# --- joint_0_z をオフセット分だけずらして torque_df に追加 ---
pos_map = pos_df.set_index("frame")["joint_0_z"]
min_map = pos_df.set_index("frame")["is_local_min_joint0_z"]
torque_df["joint_0_z"] = torque_df["frame"].map(
    lambda t: pos_map.get(t + offset, np.nan)
)
torque_df["is_local_min_joint0_z"] = torque_df["frame"].map(
    (lambda t: int(min_map.get(t + offset, False)))
)

# --- 3Dキーポイントリスト作成 ---
num_joints = 12  # CSVに含まれる関節数
kpts_list = []
for frame_idx in sorted(pos_df["frame"].unique()):
    row = pos_df[pos_df["frame"] == frame_idx].iloc[0]
    pts = [
        [row[f"joint_{j}_x"], row[f"joint_{j}_y"], row[f"joint_{j}_z"]]
        for j in range(num_joints)
    ]
    kpts_list.append(np.array(pts))

# --- Storage & Calculator 初期化 ---

storage = BodyPartDataStorage()
part_calculations = {
    "upper_arm_R": {"start": 0, "end": 2},
    "forearm_R": {"start": 2, "end": 4},
    "both_shoulder": {"start": 0, "end": 1},
    "both_hip": {"start": 6, "end": 7},
    "up_arm_l": {"start": 1, "end": 3},
    "forearm_L": {"start": 3, "end": 5},
    "upper_Leg_R": {"start": 6, "end": 8},
    "upper_Leg_L": {"start": 7, "end": 9},
}
calculators = {
    name: LinkVectorCalculator(cfg["start"], cfg["end"])
    for name, cfg in part_calculations.items()
}

# --- リンク情報の計算 & storage への追加 ---
for i in range(len(kpts_list)):
    for part, calc in calculators.items():
        res = calc.calculate_link_vectors(kpts_list, True, i, dt)
        if res[0] is None:
            continue
        storage.add_data(part, *res)

# --- 慣性モーメント I1..I6 を最初のフレームで一度だけ計算 ---
base_pts = kpts_list[0]
I1 = calculate_inertia_tensor(3, w, np.linalg.norm(base_pts[0] - base_pts[2]))  # 上腕
I2 = calculate_inertia_tensor(4, w, np.linalg.norm(base_pts[2] - base_pts[4]))  # 前腕
len_half = 0.25 * np.linalg.norm(base_pts[0] + base_pts[1] - base_pts[7] - base_pts[6])
I3 = calculate_inertia_tensor(1, w, len_half)  # 上胴体
I4 = calculate_inertia_tensor(0, w, len_half)  # 下胴体
I5 = calculate_inertia_tensor(6, w, np.linalg.norm(base_pts[9] - base_pts[7]))  # 太もも
I6 = calculate_inertia_tensor(
    7, w, np.linalg.norm(base_pts[11] - base_pts[9])
)  # 前脛骨

# --- トルク計算 & ローカル変換 & CSV追記 ---
n_torque = len(torque_df["frame"].unique())
for t_idx in range(n_torque):
    k_idx = t_idx + offset
    if k_idx >= len(kpts_list):
        break
    # 右半身の M,F,parts
    results_R = [
        calculate_M_and_F(I1, m1, storage.get_data("upper_arm_R"), g),
        calculate_M_and_F(I2, m2, storage.get_data("forearm_R"), g),
        calculate_M_and_F(
            I3,
            w,
            storage.get_data("both_shoulder"),
            g,
            add_part_data=storage.get_data("both_hip"),
            condition=1,
            Imode=3,
            Info_I3=base_pts,
        ),
        calculate_M_and_F(
            I4,
            w,
            storage.get_data("both_hip"),
            g,
            add_part_data=storage.get_data("both_shoulder"),
            Imode=4,
        ),
        calculate_M_and_F(I5, m4, storage.get_data("upper_Leg_R"), g),
    ]
    Ms_R, Fs_R, parts_R = map(list, zip(*results_R))

    # 左半身の M,F,parts
    results_L = [
        calculate_M_and_F(I1, m1, storage.get_data("up_arm_l"), g),
        calculate_M_and_F(I2, m2, storage.get_data("forearm_L"), g),
        calculate_M_and_F(
            I3,
            w,
            storage.get_data("both_shoulder"),
            g,
            add_part_data=storage.get_data("both_hip"),
            condition=0,
            Imode=3,
            Info_I3=base_pts,
        ),
        results_R[3],  # 下胴体は右半身で計算済みと同じ
        calculate_M_and_F(I5, m4, storage.get_data("upper_Leg_L"), g),
    ]
    Ms_L, Fs_L, parts_L = map(list, zip(*results_L))

    # 外力/外部トルク/作用点
    vec = kpts_list[k_idx][8] - kpts_list[k_idx][6]
    angle = np.degrees(np.pi / 2 - np.arccos(vec[2] / np.linalg.norm(vec)))
    f_E = (
        np.array([0, 0, w * 0.66 * np.linalg.norm(g) / 2])
        if abs(angle) < 20
        else np.zeros(3)
    )
    tau_E = np.zeros(3)
    r_x = storage.get_data("both_hip")[k_idx]["centroid"]

    # 重心リスト r_gs / r_gsL を短縮リスト内包表現で
    def mix(i):
        return (
            0.75 * storage.get_data("both_shoulder")[k_idx]["centroid"]
            + 0.25 * storage.get_data("both_hip")[k_idx]["centroid"]
        )

    def mixL(i):
        return (
            0.25 * storage.get_data("both_shoulder")[k_idx]["centroid"]
            + 0.75 * storage.get_data("both_hip")[k_idx]["centroid"]
        )

    r_gs = [
        (
            storage.get_data(part)[k_idx]["centroid"]
            if part not in ("both_shoulder", "both_hip")
            else mix(None)
        )
        for part in parts_R
    ]
    r_gsL = [
        (
            storage.get_data(part)[k_idx]["centroid"]
            if part not in ("both_shoulder", "both_hip")
            else mixL(None)
        )
        for part in parts_L
    ]

    torques_R = calculate_individual_torques(
        Ms_R, Fs_R, np.array(r_gs), tau_E, f_E, r_x, parts_R, storage
    )
    torques_L = calculate_individual_torques(
        Ms_L, Fs_L, np.array(r_gsL), tau_E, f_E, r_x, parts_L, storage
    )

    # CSVへ出力 (重複回避のためサフィックス付与)
    for tau_gl, part in torques_R:
        suffix = "_R" if part in ("both_shoulder", "both_hip") else ""
        tau_loc = compute_local_torque(
            tau_gl, storage.get_data(part)[k_idx]["relative_position_vector"]
        )
        for ax, val in zip(("x", "y", "z"), tau_loc):
            torque_df.loc[
                torque_df["frame"] == t_idx, f"{part}{suffix}_torque_local_{ax}"
            ] = val
    for tau_gl, part in torques_L:
        suffix = "_L" if part in ("both_shoulder", "both_hip") else ""
        tau_loc = compute_local_torque(
            tau_gl, storage.get_data(part)[k_idx]["relative_position_vector"]
        )
        for ax, val in zip(("x", "y", "z"), tau_loc):
            torque_df.loc[
                torque_df["frame"] == t_idx, f"{part}{suffix}_torque_local_{ax}"
            ] = val

# --- ver2 保存 ---
root, ext = os.path.splitext(torque_csv_path)
out_path = root + "_ver2" + ext
torque_df.to_csv(out_path, index=False)
print(f"✅ 保存完了: {out_path}")
