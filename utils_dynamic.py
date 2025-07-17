import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=no-member
import cv2 as cv
import pandas as pd
from config import w, folder_path


def calculate_inertia_tensor(k, w, l):
    """
    質量と長さに基づいて、対象部位の慣性テンソルを計算する関数。

    CSVファイルから部位ごとの推定係数（a, b, c）を読み込み、以下の式を用いて
    各軸方向（x, y, z）の慣性モーメントを計算し、対角慣性テンソルを構築する：

        I = a * w + b * l + c

    Parameters
    ----------
    k : int
        係数データを取得するCSVの行インデックス（部位を表す）。
    w : float
        部位の質量（kg）。
    l : float
        部位の長さ（m）。

    Returns
    -------
    I_tensor : ndarray
        3x3の対角慣性テンソル（慣性行列）。

    Notes
    -----
    - CSVファイルのパスは `folder_path` に依存。
    - CSVファイル名は "Moment of inertia estimation coefficient boys.csv" 固定。
    - 対角行列の形でのみ出力され、オフダイアゴナル要素（慣性積）は常に0とされる。
    """

    csv_path = folder_path + "\\Moment of inertia estimation coefficient boys.csv"

    data = pd.read_csv(csv_path)
    print("Data loaded")
    row = data.iloc[k]
    # 位置指定で取り出す
    a_x, b_x, c_x = row.iloc[1], row.iloc[2], row.iloc[3]
    a_y, b_y, c_y = row.iloc[4], row.iloc[5], row.iloc[6]
    a_z, b_z, c_z = row.iloc[7], row.iloc[8], row.iloc[9]
    # 慣性モーメントの計算
    I_x = a_x * w + b_x * l + c_x
    I_y = a_y * w + b_y * l + c_y
    I_z = a_z * w + b_z * l + c_z
    I_tensor = np.array([[I_x, 0, 0], [0, I_y, 0], [0, 0, I_z]])
    return I_tensor


def skew_symmetric_matrix(v):
    """ベクトルからスキュー対称行列を生成する"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def calculate_M_and_F(
    I, m, part_data, g, add_part_data=None, condition=None, Imode=None, Info_I3=None
):
    """
    慣性テンソルと角運動量に基づいて、回転の慣性力（M）および並進の慣性力(F）を計算する関数。

    この関数は、指定された部位の運動データから角加速度や角速度、重心加速度を取得し、
    慣性テンソルと質量を用いて、モーメント（M）と力（F）を計算する。また、特定の部位や
    条件に応じて補正処理も行う。

    Parameters
    ----------
    I : ndarray
        慣性テンソル（3x3行列）。
    m : float
        質量（条件により再計算される場合あり）。
    part_data : list of dict
        部位ごとの物理量（"omega", "dot_omega", "dot_dot_pg", "part_name" を含む辞書）のリスト。
    g : ndarray
        重力加速度ベクトル。
    add_part_data : list of dict, optional
        追加の部位データ（主に肩の補正に用いる）。
    condition : int, optional
        肩の左右を示すフラグ（1: 右肩、0: 左肩）。

    Returns
    -------
    M : ndarray
        モーメントベクトル。
    F : ndarray
        力ベクトル。
    part_name : str
        計算対象部位の名前。

    Notes
    -----
    - `I3` または `I4` に一致する場合、肩の補正処理を実行。
    - `Info_I3`, `w`, `I3`, `I4` はグローバルスコープで定義されている必要がある。
    - `omega` および `dot_omega` がゼロベクトルに設定される場合もある（特に I4 の場合）。
    """

    omega = part_data[-1]["omega"]
    dot_omega = part_data[-1]["dot_omega"]
    dot_dot_pg = part_data[-1]["dot_dot_pg"]
    part_name = part_data[-1]["part_name"]  # 部位名の取り出し

    if Imode == 3:
        A1 = np.linalg.norm((Info_I3[1][:2] + Info_I3[0][:2]) * 0.5 - Info_I3[5][:2])
        A0 = np.linalg.norm((Info_I3[1][:2] + Info_I3[0][:2]) * 0.5 - Info_I3[4][:2])
        dot_dot_pg = (
            part_data[-1]["dot_dot_pg"] * 3 + add_part_data[-1]["dot_dot_pg"]
        ) * 0.25

        if condition == 1:
            # 右肩1
            m = w * 0.276 * A0 / (A0 + A1)
            omega = omega * (-1)
            dot_omega = dot_omega * (-1)
        elif condition == 0:  # 左肩0
            m = w * 0.276 * A1 / (A0 + A1)
    elif Imode == 4:
        omega = np.array([0, 0, 0])
        dot_omega = np.array([0, 0, 0])
        dot_dot_pg = (
            part_data[-1]["dot_dot_pg"] * 3 + add_part_data[-1]["dot_dot_pg"]
        ) / 4

    M = I.dot(dot_omega) + np.cross(omega, I.dot(omega))
    F = m * (dot_dot_pg - g)

    return M, F, part_name

    # 個々のトルクを計算する関数


def calculate_individual_torques(Ms, Fs, r_gs, tau_E, f_E, r_x, parts, storage):
    """
    各身体部位にかかる関節トルクを運動連鎖に沿って再帰的に計算する関数。

    Parameters
    ----------
    Ms : list of ndarray, shape (3,)
        各部位における回転モーメントベクトル M_i のリスト。
    Fs : list of ndarray, shape (3,)
        各部位にかかる並進力ベクトル F_i のリスト。
    r_gs : list of ndarray, shape (3,)
        各部位の重心位置ベクトル r^{g_i} のリスト。Ms, Fs, parts と同じ順序で並べること。
    tau_E : ndarray, shape (3,)
        外部トルクベクトル。
    f_E : ndarray, shape (3,)
        外力ベクトル。
    r_x : ndarray, shape (3,)
        外力作用点の位置ベクトル。
    parts : list of str
        各部位の名前リスト。storage.get_data(part)[-1] がその部位データを返す順序に合わせる。
    storage : BodyPartDataStorage
        各部位の p1（関節位置）や重心位置などを保持しているインスタンス。

    Returns
    -------
    torques : list of tuple
        (tau_j, part_name) のリスト。tau_j は部位 j にかかるトルクベクトル、part_name は部位名。
    """
    torques = []
    n = len(Ms)
    for j in range(n):
        part_j = parts[j]
        data_j = storage.get_data(part_j)[-1]
        p1 = data_j["p1"]  # 関節位置

        # 1) 回転モーメントの合計
        sum_M = np.sum(Ms[j:], axis=0)
        # if not np.all(np.isfinite(sum_M)):
        # print(f"NaN detected in sum_M for {part_j}. Setting to zero.")
        # 2) 並進力によるモーメントの合計
        sum_F = np.zeros(3)
        for i in range(j, n):
            # r_j^{g_i} = 重心 i から関節 j までのベクトル
            r_ji = r_gs[i] - p1
            sum_F += skew_symmetric_matrix(r_ji) @ Fs[i]

        # 3) 外部トルク・外力のモーメント
        tau_x_fE = skew_symmetric_matrix(r_x - p1) @ f_E

        # トルク合成
        tau_j = sum_M + sum_F - tau_E - tau_x_fE

        # NaN 対策
        if not np.all(np.isfinite(tau_j)):
            tau_j = np.zeros(3)
            # print(f"NaN detected in torque calculation for {part_j}. Setting to zero.")

        torques.append((tau_j, part_j))

    return torques


def update_graphs(new_data_points, lines, axes, torque_sss):
    """
    時系列グラフをリアルタイムで更新するための関数。

    各プロットに新しいデータポイントを追加し、最大100点まで保持するように
    古いデータを削除しながら、対応するMatplotlibのラインと軸を更新する。

    Parameters
    ----------
    new_data_points : list of float
        各グラフに追加する新しいデータ点（各ラインに1つずつ対応）。
    lines : list of Line2D
        Matplotlibの折れ線グラフオブジェクト（`ax.plot()` などで生成されたもの）。
    axes : ndarray of Axes
        グラフを描画しているMatplotlibのAxesオブジェクトの配列（flatten()して利用）。
    torque_sss : list of list
        各ラインに対応するyデータの履歴（最大100個まで保持）。

    Notes
    -----
    - `torque_sss` は `lines` に対応する y データの生配列（データ履歴）です。
    - 各折れ線は最大100点まで描画され、それ以上のデータは先頭から削除されます。
    - `plt.draw()` と `plt.pause()` によってグラフがインタラクティブに更新されます。
    """

    for new_data, (line, ax), y in zip(
        new_data_points, zip(lines, axes.flatten()), torque_sss
    ):
        # 新しいデータポイントを追加
        y.append(new_data)
        if len(y) > 100:
            y.pop(0)  # リストが100を超えたら最初の要素を削除

        line.set_ydata(y)  # 折れ線グラフを更新
        ax.relim()  # データ範囲を更新
        ax.autoscale_view()  # 軸を再スケーリング

    plt.draw()
    plt.pause(0.01)


def draw_rotated_rectangle(
    frame, OA, OB, color, alpha=0.8, AC_width=10, shoulder_mode=False
):
    """指定された座標で長方形を描画し、フレームに適用する関数"""
    AB = OB - OA
    AC = (
        np.array([-AB[1], AB[0]]) / np.linalg.norm(np.array([-AB[1], AB[0]])) * AC_width
    )  # ABに垂直なベクトル

    if shoulder_mode:
        rotated_coords = np.array(
            [
                OA - 2 * AC,  # 左上の点
                OB - 2 * AC,  # 右上の点
                OB,  # 右下の点
                OA,  # 左下の点
            ],
            dtype=np.int32,
        )
    # 回転後の座標
    else:
        rotated_coords = np.array(
            [
                OA - AC,  # 左上の点
                OB - AC,  # 右上の点
                OB + AC,  # 右下の点
                OA + AC,  # 左下の点
            ],
            dtype=np.int32,
        )

    overlay = frame.copy()
    cv.fillPoly(overlay, [rotated_coords], color)
    return cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def integrate_values_with_initial(dt, a, current_value):
    """
    瞬間値のリスト、初期積分値のリスト、および時間間隔を受け取り、積分した結果のリストを返す。

    :param dt: float - 各ステップの時間間隔
    :param a_values: list of float - 各ステップにおける瞬間値（例えば加速度）
    :param initial_values: list of float - 積分の初期値（例えば初期速度や初期位置）
    :return: list of float - 積分値（例えば速度や位置）
    """
    current_value += a * dt  # 瞬間値を積分

    return current_value


# インパルス計算関数（compute_impulse と同一）
def compute_impulse(series: pd.Series, dt: float):
    arr = series.to_numpy()
    pos_imp = arr[arr > 0].sum() * dt
    neg_imp = arr[arr < 0].sum() * dt
    return pos_imp, neg_imp
