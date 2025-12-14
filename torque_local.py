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
    if not np.all(np.isfinite(link_vec)):
        return torque_global
    norm_link = np.linalg.norm(link_vec)
    if norm_link < 1e-12:
        return torque_global

    z_axis = link_vec / norm_link

    reference_axes = (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    x_axis = None
    for ref in reference_axes:
        if abs(np.dot(z_axis, ref)) >= 0.95:
            continue
        candidate = np.cross(ref, z_axis)
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm < 1e-12:
            continue
        x_axis = candidate / candidate_norm
        break

    if x_axis is None:
        return torque_global

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-12:
        return torque_global
    y_axis /= y_norm

    rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
    torque_local = rotation.T @ torque_global
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
