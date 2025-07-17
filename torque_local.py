import pandas as pd
import numpy as np
from pathlib import Path

# ファイルパスの設定
kpts3d_path = Path("/mnt/data/kpts3d_0407_162237 - コピー.csv")
torque_path = Path("/mnt/data/aim_torque_vec_0407_162237 - コピー.csv")

# データの読み込み
kpts_df = pd.read_csv(kpts3d_path)
torque_df = pd.read_csv(torque_path)

# 必要な関節（右肘: joint3, 右手: joint5）の座標を抽出
elbow_cols = ["joint_3_x", "joint_3_y", "joint_3_z"]
wrist_cols = ["joint_5_x", "joint_5_y", "joint_5_z"]

elbow = kpts_df[elbow_cols].values
wrist = kpts_df[wrist_cols].values

# トルクの対象列（wristのトルク）
torque_cols = ["wrist_x", "wrist_y", "wrist_z"]
torque_vals = torque_df[torque_cols].values


# ローカル座標系に変換する関数
def compute_local_torque(torque_global, link_vec):
    l_x, l_y, l_z = link_vec
    phi = np.arctan2(l_y, l_x)  # z軸回転角
    theta = np.arctan2(np.sqrt(l_x**2 + l_y**2), l_z)  # y軸回転角

    # z軸回転
    Rz = np.array(
        [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )

    # y軸回転
    Ry = np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )

    R = Ry @ Rz  # 合成回転行列
    torque_local = R @ torque_global
    return torque_local


# ローカルトルクに変換
local_torques = []
for i in range(len(torque_vals)):
    link_vec = wrist[i] - elbow[i]
    local_tau = compute_local_torque(torque_vals[i], link_vec)
    local_torques.append(local_tau)

local_torque_array = np.array(local_torques)

# 結果をDataFrameにして保存用にマージ
local_torque_df = torque_df.copy()
local_torque_df["wrist_local_x"] = local_torque_array[:, 0]
local_torque_df["wrist_local_y"] = local_torque_array[:, 1]
local_torque_df["wrist_local_z"] = local_torque_array[:, 2]
