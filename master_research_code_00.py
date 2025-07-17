# %%
import logging
import os
import time
import winsound

# pylint: disable=no-member
import cv2 as cv
import japanize_matplotlib  # pylint: disable=unused-import # 日本語表示のサポート
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

from body_part_storage_module import BodyPartDataStorage
from config import (
    PADDING,
    dt,
    folder_path,
    fps,
    frame_shape,
    g,
    input_stream1,
    input_stream2,
    m1,
    m2,
    m4,
    pose_keypoints,
    rm_path,
    save_dir,
    timestamp,
    w,
)
from link_vector_calculator_module import LinkVectorCalculator
from utils import DLT, extract_keypoints, get_projection_matrix, put_text_jp
from utils_dynamic import (
    calculate_individual_torques,
    calculate_inertia_tensor,
    calculate_M_and_F,
    draw_rotated_rectangle,
    update_graphs,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 姿勢推定のためのdetectorオブジェクトを作成
pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("✅ Mediapipe・モデル準備 完了")
# %%

d = 100
j = 0

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logging.debug("This message should go to the log file.")
kpt_3d = []
torques = []
cv.namedWindow("MyWindow", cv.WINDOW_NORMAL)

# CSVファイルを読み込む
df_rm = pd.read_csv(rm_path)
# ウィンドウの位置を設定 (x=100, y=100)

cv.moveWindow("MyWindow", 0, 0)

THRESHOLD = None


# matplotlibのプロットをインタラクティブモードで使用
plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2のグリッドでサブプロットを作成

# 初期データセットとラインオブジェクトのリスト
torque_sss = [list(np.random.randn(100)) for _ in range(4)]
colors = ["red", "blue", "green", "black"]
labels = ["右手首", "右肘", "右肩", "体幹"]
lines = [
    ax.plot(np.linspace(-10, 0, 100), y, color=color, label=label)[0]
    for y, ax, color, label in zip(torque_sss, axes.flatten(), colors, labels)
]

for ax in axes.flatten():
    ax.set_xlim(-10, 0)  # X軸の範囲を-10から0まで設定
    # ax.set_ylim(0, 15000)  # X軸の範囲を-10から0まで設定

    ax.set_xlabel("時間(秒)")  # X軸ラベル
    ax.set_ylabel("トルク値")  # Y軸ラベル
    ax.legend()  # 凡例を表示

plt.show()

# 2つのブール変数
byte_vData = False
byte_AccData = False

aim_torque = []
AIM_count = 0

# print("Constants defined")
# 慣性モーメントテンソル（ダミー値で初期化）

true_count = 0
false_count = 0

AIM_bool = False
I1 = I2 = I3 = I4 = I5 = I6 = I7 = None

storage = BodyPartDataStorage()
# 部位ごとの計算設定を辞書に格納
part_calculations = {
    "upper_arm_R": {"start": 3, "end": 1},
    "forearm_R": {"start": 5, "end": 3},
    "both_shoulder": {"start": 0, "end": 1},
    "both_hip": {"start": 6, "end": 7},
    "up_arm_l": {"start": 2, "end": 0},
    "forearm_L": {"start": 4, "end": 2},
    "upper_Leg_R": {"start": 7, "end": 9},
    "upper_Leg_L": {"start": 6, "end": 8},
}
# LinkVectorCalculatorのインスタンスを辞書に保持
calculators = {}
print("Calculators created")
# 各部位に対してインスタンスを作成し、辞書に格納
for part, settings in part_calculations.items():
    calculators[part] = LinkVectorCalculator(settings["start"], settings["end"])

print("Video loaded")
# put camera id as command line arguements

if input_stream1 == 0 and input_stream2 == 1:
    file_mode = False
else:
    file_mode = True
reps = 10
one_rm_percentage = df_rm.loc[df_rm["反復回数"] == reps, "1RM%"].values[0] / 100
# ファイルを読み込みモードで開く
with open(folder_path + "\\max_value.txt", "r") as file:
    # ファイルからデータを一行読み込む
    data = file.readline().strip()  # strip()で余計な空白や改行を除去
    if data == "":
        print("No data in file")
    else:
        # 読み込んだデータをfloatに変換#X
        max_value = float(data)
        #    print(f"ファイル内の最大値: {max_value}")
        THRESHOLD = max_value / one_rm_percentage
# get projection matrices
# --- ユーザーに「準備OKで再開してね」と促す ---

P0 = get_projection_matrix(0, file_mode)
P1 = get_projection_matrix(1, file_mode)
print("Projection matrices loaded")
# %%
save_path0 = f"cam0_output_{timestamp}.mp4"
save_path1 = f"cam1_output_{timestamp}.mp4"

cap0 = cv.VideoCapture(input_stream1)  # ビデオファイル1
cap1 = cv.VideoCapture(input_stream2)  # ビデオファイル2

# 最初のフレームを読んで、サイズを取得
ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

if not ret0 or not ret1:
    print("❌ 動画ファイルの読み込みに失敗しました")
    exit()

h0, w0 = frame0.shape[:2]
h1, w1 = frame1.shape[:2]

# FPS（必要なら動画から取得する）


# 保存用VideoWriterの初期化（frameの実サイズに合わせる）
fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer0 = cv.VideoWriter(save_path0, fourcc, fps, (w0, h0))
writer1 = cv.VideoWriter(save_path1, fourcc, fps, (w1, h1))

# 動作チェック
if not writer0.isOpened():
    print(f"❌ writer0 の初期化に失敗しました（サイズ: {w0}x{h0}）")
if not writer1.isOpened():
    print(f"❌ writer1 の初期化に失敗しました（サイズ: {w1}x{h1}）")


kpts_3d = []  # 3Dキーポイントデータを格納するリスト
kpts_cam0 = []
kpts_cam1 = []
maxs = [0, 0, 0, 0]
SKIP_FRAMES = 0
WHILE_COUNT = 0
print("Starting loop")
winsound.Beep(500, 1000)  # 周波数と持続時間を指定して音を鳴らす（例えば、500Hzで1秒間）

# %%
while True:
    start_time = time.perf_counter()
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if SKIP_FRAMES % 5 == 0:
        SKIP_FRAMES += 1

    else:
        SKIP_FRAMES += 1
        continue

    if not ret0 or not ret1:
        print("Video ended")
        break
        # 保存処理（BGRのままでOK）
    writer0.write(frame0)
    writer1.write(frame1)
    if frame0.shape[1] != 720:
        frame0 = frame0[
            :,
            frame_shape[1] // 2
            - frame_shape[0] // 2 : frame_shape[1] // 2
            + frame_shape[0] // 2,
        ]
        frame1 = frame1[
            :,
            frame_shape[1] // 2
            - frame_shape[0] // 2 : frame_shape[1] // 2
            + frame_shape[0] // 2,
        ]

    frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    frame0.flags.writeable = False
    frame1.flags.writeable = False
    results0 = pose0.process(frame0)
    results1 = pose1.process(frame1)
    frame0.flags.writeable = True
    frame1.flags.writeable = True
    frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
    frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
    frame0_keypoints, frame1_keypoints = extract_keypoints(
        results0, results1, pose_keypoints, frame0, frame1
    )

    frame_p3ds = []

    for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        if uv1[0] == -1 or uv2[0] == -1:
            _p3d = [-1, -1, -1]
        else:
            _p3d = DLT(P0, P1, uv1, uv2)

        frame_p3ds.append(_p3d)
    temp_np = np.array(frame_p3ds).reshape((12, 3)) * 0.01
    transformed_p3ds = np.zeros_like(temp_np)
    if file_mode:
        transformed_p3ds[:, 0] = -temp_np[:, 0]  # 0番目の要素の符号を逆に
        transformed_p3ds[:, 1] = -temp_np[:, 2]  # 2番目の要素の符号を逆にして1番目に
        transformed_p3ds[:, 2] = -temp_np[:, 1]  # 1番目の要素を2番目に

    else:
        transformed_p3ds[:, 0] = -temp_np[:, 0]  # 0番目の要素の符号を逆に
        transformed_p3ds[:, 1] = -temp_np[:, 2]  # 2番目の要素の符号を逆にして1番目に
        transformed_p3ds[:, 2] = -temp_np[:, 1]  # 1番目の要素を2番目に
    kpts_3d.append(transformed_p3ds)
    for part_name, calculator in calculators.items():
        # データ計算
        # print(part_name)
        if file_mode:
            i = len(kpts_3d) - 1
        else:
            i = 0
        result = calculator.calculate_link_vectors(kpts_3d, file_mode, i, dt)

        # 結果が不完全ならスキップ
        if result[0] is None:
            continue

        # 結果を7つに展開
        r_vec, vel_vec, omega, cent, p1, acc, ang_acc = result

        # print("calculated")
        # データ格納
        storage.add_data(part_name, r_vec, vel_vec, omega, cent, p1, ang_acc, acc)

    upper_arm_R_data = storage.get_data("upper_arm_R")
    # print("Data stored"+upper_arm_R_data[-1]['part_name'])
    forearm_R_data = storage.get_data("forearm_R")
    both_shoulder_data = storage.get_data("both_shoulder")
    both_hip_data = storage.get_data("both_hip")

    upper_armL_data = storage.get_data("up_arm_l")
    forearm_L_data = storage.get_data("forearm_L")
    upper_LEG_R_data = storage.get_data("upper_Leg_R")
    upper_LEG_L_data = storage.get_data("upper_Leg_L")

    if len(kpts_3d) < 4:
        if len(kpts_3d) == 3:
            I1 = calculate_inertia_tensor(
                3, w, np.linalg.norm(transformed_p3ds[0] - transformed_p3ds[2])
            )  # 上腕
            I2 = calculate_inertia_tensor(
                4, w, np.linalg.norm(transformed_p3ds[2] - transformed_p3ds[4])
            )  # 前腕
            len_half_body = 0.25 * np.linalg.norm(
                transformed_p3ds[0]
                + transformed_p3ds[1]
                - transformed_p3ds[7]
                - transformed_p3ds[6]
            )
            I3 = calculate_inertia_tensor(1, w, len_half_body)  # 上胴体
            I4 = calculate_inertia_tensor(0, w, len_half_body)  # 下胴体
            I5 = calculate_inertia_tensor(
                6, w, np.linalg.norm(transformed_p3ds[9] - transformed_p3ds[7])
            )  # 太もも
            I6 = calculate_inertia_tensor(
                7, w, np.linalg.norm(transformed_p3ds[11] - transformed_p3ds[9])
            )  # 前足
            I7 = calculate_inertia_tensor(2, w, 0.25)  # 頭
        continue

    # 計算とデータの格納をループで行う
    if len(kpts_3d) < 7:
        continue
    # トルクと力の計算を関数を使って実行
    M1, F1, Part1 = calculate_M_and_F(I1, m1, upper_arm_R_data, g)
    M2, F2, Part2 = calculate_M_and_F(I2, m2, forearm_R_data, g)
    M3L, F3L, Part3L = calculate_M_and_F(
        I3,
        w,
        both_shoulder_data,
        g,
        add_part_data=both_hip_data,
        condition=0,
        Imode=3,
        Info_I3=transformed_p3ds,
    )
    M3, F3, Part3 = calculate_M_and_F(
        I3,
        w,
        both_shoulder_data,
        g,
        add_part_data=both_hip_data,
        condition=1,
        Imode=3,
        Info_I3=transformed_p3ds,
    )
    M4, F4, Part4 = calculate_M_and_F(
        I4, w, both_hip_data, g, add_part_data=both_shoulder_data, Imode=4
    )
    M5, F5, Part5 = calculate_M_and_F(I5, m4, upper_LEG_R_data, g)
    M1L, F1L, Part1L = calculate_M_and_F(I1, m1, upper_armL_data, g)
    M2L, F2L, Part2L = calculate_M_and_F(I2, m2, forearm_L_data, g)
    # M6, F6, Part6 = calculate_M_and_F(I5, w, upper_LEG_R_data, g)
    M5L, F5L, Part5L = calculate_M_and_F(I5, m4, upper_LEG_L_data, g)

    # 既に与えられている3次元ベクトルの配列
    # Ms = [M5,M4,M3,M2,M1]
    # Fs = [F5,F4,F3,F2,F1]
    # parts = [Part5,Part4,Part3,Part2,Part1]
    Ms = [M1, M2, M3, M4, M5]
    Fs = [F1, F2, F3, F4, F5]
    parts = [Part1, Part2, Part3, Part4, Part5]

    vector = transformed_p3ds[8] - transformed_p3ds[6]
    angle_with_xy_plane = np.pi / 2 - np.arccos(vector[2] / np.linalg.norm(vector))
    angle_degrees = np.degrees(angle_with_xy_plane)

    print("angle=", angle_degrees)
    if (angle_degrees > 20) and (angle_degrees > -20):
        f_E = np.array([0, 0, w * 0.66 * np.linalg.norm(g) / 2])
    else:
        f_E = np.array([0, 0, 0])

    if file_mode:
        # f_E = np.array([0, 0, w*0.66*np.linalg.norm(g)/2])
        f_E = np.array([0, 0, 0])
    r_x = both_hip_data[-1]["centroid"]
    # r_x = .5*(both_shoulder_data[-1]['p1']+both_hip_data[-1]['p1'])+.25*(both_shoulder_data[-1]['relative_position_vector']+both_hip_data[-1]['relative_position_vector'])  # この例での r_x

    tau_E = np.array([0, 0, 0])

    r_g = [
        forearm_R_data[-1]["centroid"],
        upper_arm_R_data[-1]["centroid"],
        (both_shoulder_data[-1]["centroid"] * 3 + both_hip_data[-1]["centroid"]) / 4,
        (both_shoulder_data[-1]["centroid"] + both_hip_data[-1]["centroid"] * 3) / 4,
        upper_LEG_R_data[-1]["centroid"],
    ]

    torques = calculate_individual_torques(
        Ms, Fs, np.array(r_g), tau_E, f_E, r_x, parts, storage
    )
    # ディクショナリにトルク値を格納
    for torque, part in torques:
        storage.add_torque(part, torque)
    # torque_sssへのトルク値の追加
    temp_torques = [torques[i][0] for i in range(len(torques))]
    aim_torque.append(temp_torques)
    # print("左手首:",int(torques[0][0]),"\t左肘:",int(torques[1][0]),"\t左肩:",int(torques[2][0]),"\t体幹?:",int(torques[3][0]),)
    # 結果の表示

    # cv.imshow('cam1', frame1)
    # 新しい画像サイズを計算（余白分を加える）
    new_height = frame0.shape[0]
    new_width = frame0.shape[1] + PADDING

    frame0 = draw_rotated_rectangle(
        frame0,
        np.array(frame0_keypoints[0]),
        np.array(frame0_keypoints[2]),
        (0, 0, int(np.clip(np.linalg.norm(torques[1][0]) / 140, 0, 1) * 255)),
        alpha=0.8,
    )
    frame0 = draw_rotated_rectangle(
        frame0,
        np.array(frame0_keypoints[2]),
        np.array(frame0_keypoints[4]),
        (0, 0, int(np.clip(np.linalg.norm(torques[0][0]) / 140, 0, 1) * 255)),
        alpha=0.8,
    )
    frame0 = draw_rotated_rectangle(
        frame0,
        np.array(frame0_keypoints[0]),
        (np.array(frame0_keypoints[1]) + np.array(frame0_keypoints[0])) * 0.5,
        (0, 0, int(np.clip(np.linalg.norm(torques[2][0]) / 140, 0, 1) * 255)),
        AC_width=30,
        alpha=0.8,
        shoulder_mode=True,
    )
    frame0 = draw_rotated_rectangle(
        frame0,
        np.array(frame0_keypoints[1]),
        np.array(frame0_keypoints[3]),
        (0, 0, int(np.clip(np.linalg.norm(torques[1][0]) / 140, 0, 1) * 255)),
        alpha=0.8,
    )
    frame0 = draw_rotated_rectangle(
        frame0,
        np.array(frame0_keypoints[3]),
        np.array(frame0_keypoints[5]),
        (0, 0, int(np.clip(np.linalg.norm(torques[0][0]) / 140, 0, 1) * 255)),
        alpha=0.8,
    )
    frame0 = draw_rotated_rectangle(
        frame0,
        (np.array(frame0_keypoints[1]) + np.array(frame0_keypoints[0])) * 0.5,
        np.array(frame0_keypoints[1]),
        (0, 0, int(np.clip(np.linalg.norm(torques[2][0]) / 140, 0, 1) * 255)),
        AC_width=30,
        alpha=0.8,
        shoulder_mode=True,
    )

    # 新しい画像データ（余白付き）を作成
    new_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_frame[: frame0.shape[0], : frame0.shape[1]] = (
        frame0  # 元の画像を新しい画像データにコピー
    )
    # 余白部分に文字を描画

    # text5 = ("到達回数"+str(AIM_count)+"/5")
    color = (255, 255, 255)  # 白色
    text = "左手首:" + str(int(np.linalg.norm(torques[0][0])))
    text1 = "左肘:" + str(int(np.linalg.norm(torques[1][0])))
    text2 = "左肩:" + str(int(np.linalg.norm(torques[2][0])))
    text3 = "体幹?:" + str(int(np.linalg.norm(torques[3][0])))
    font = cv.FONT_HERSHEY_SIMPLEX
    new_frame = put_text_jp(new_frame, text, (new_width - 300, 40), 24, color, 20)
    new_frame = put_text_jp(
        new_frame, "終了したい時はQキー長押し", (new_width - 300, 10), 24, color, 20
    )
    new_frame = put_text_jp(new_frame, text1, (new_width - 300, 70), 24, color, 20)
    new_frame = put_text_jp(new_frame, text2, (new_width - 300, 100), 24, color, 20)
    new_frame = put_text_jp(new_frame, text3, (new_width - 300, 130), 24, color, 20)
    if THRESHOLD is not None:
        text5 = "到達回数" + str(AIM_count) + "/5"
        new_frame = put_text_jp(new_frame, text5, (new_width - 300, 190), 24, color, 20)
    # font_scale = 1
    # thickness = 2
    # cv.putText(new_frame, text, (10, new_height - 50), font, font_scale, color, thickness)
    update_graphs(
        (
            int(np.linalg.norm(torques[0][0])),
            int(np.linalg.norm(torques[1][0])),
            int(np.linalg.norm(torques[2][0])),
            int(np.linalg.norm(torques[3][0])),
        ),
        lines,
        axes,
        torque_sss,
    )

    # ウィンドウのサイズをリサイズ（ウィンドウを画像サイズに合わせる）
    cv.resizeWindow("MyWindow", new_width, new_height)
    cv.imshow("MyWindow", new_frame)
    # cv.imshow('cam1', frame1)
    WHILE_COUNT += 1
    end_time = time.perf_counter()
    dt = end_time - start_time
    print("dt=", f"{dt:.3f}")
    print("FPS:", 1 / dt)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# %%
cv.destroyAllWindows()

cap0.release()
cap1.release()
# if aim_torque:
print("max value", maxs)
writer0.release()
writer1.release()

# ファイルに書き込む
# with open(folder_path + "\\max_value.txt", "w", encoding="utf-8") as file:
#    file.write(str(max(aim_torque)))

# -------------------------------
# ① kpts_3d：3D座標データを保存
# -------------------------------
# kpts_3d: List[np.ndarray]  → (12, 3)の各フレームごとのリストと想定
flattened_rows = []

for frame_idx, frame in enumerate(kpts_3d):
    row = {"frame": frame_idx}
    for joint_idx in range(frame.shape[0]):
        x, y, z = frame[joint_idx]
        row[f"joint_{joint_idx}_x"] = round(x, 4)
        row[f"joint_{joint_idx}_y"] = round(y, 4)
        row[f"joint_{joint_idx}_z"] = round(z, 4)
    flattened_rows.append(row)

df_coords = pd.DataFrame(flattened_rows)
df_coords.to_csv(os.path.join(save_dir, f"kpts3d_{timestamp}.csv"), index=False)
print("✅ 3D座標データを保存しました")

# -------------------------------
# ② トルクデータを保存
# -------------------------------
# torques: [(値, 部位名)] × 各フレーム分

# 部位名の順番（torques[i][1] の順に合わせておくこと）
part_names = ["wrist", "elbow", "shoulder", "core", "hip"]

# CSVに出力するデータを整形
csv_rows = []
for frame_idx, frame_data in enumerate(aim_torque):
    row = {"frame": frame_idx}
    for part_idx, vec in enumerate(frame_data):
        part = part_names[part_idx]
        row[f"{part}_x"] = round(vec[0], 4)
        row[f"{part}_y"] = round(vec[1], 4)
        row[f"{part}_z"] = round(vec[2], 4)
    csv_rows.append(row)

# DataFrameにして保存
df = pd.DataFrame(csv_rows)
save_path = os.path.join(save_dir, f"aim_torque_vec_{timestamp}.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"✅ aim_torque（ベクトル形式）を保存しました: {save_path}")
# %%
