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
import csv
import pygame
import json

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
    SKIP_FRAMES,
    WHILE_COUNT,
    z_value,
    cycle_switch,
    part_keys,
    impulse_records,
    current_torque_history,
    prev_cycle_frame,
    min_history_len,
    detector,
    gauge,
    current_impulses,
)
from link_vector_calculator_module import LinkVectorCalculator
from utils import (
    DLT,
    extract_keypoints,
    get_projection_matrix,
    put_text_jp,
    compute_local_torque,
    PushCycleDetector,
)
from utils_dynamic import (
    calculate_individual_torques,
    calculate_inertia_tensor,
    calculate_M_and_F,
    draw_rotated_rectangle,
    update_graphs,
    compute_impulse,
)
from Gauge_display import GaugeDisplay

# ── BGMファイル定義 ───────────────────────────
BGM_NORMAL = os.path.join(os.path.dirname(__file__), "union.mp3")
BGM_ALERT = os.path.join(os.path.dirname(__file__), "recover.mp3")

# pygame.mixer 初期化
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# 現在流しているBGM状態を管理するフラグ
# "normal" → 通常BGM,  "alert" → 切替後のBGM
current_bgm = "normal"

START_OFFSET = 7.0  # 再生開始位置 [秒]


def play_bgm(state: str, fade_ms: int = 1000, start_sec: float = 0.0):
    """
    state に応じたBGMをフェードイン付きで再生し、
    再生開始2秒後からスタートさせる。
    """
    path = BGM_NORMAL if state == "normal" else BGM_ALERT
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        # ループ再生＆フェードイン
        pygame.mixer.music.play(loops=-1, fade_ms=fade_ms)
        # 再生開始位置を秒単位で設定（mp3でも動作します）
        pygame.mixer.music.set_pos(start_sec)
        print(
            f"▶️ BGM再生: {os.path.basename(path)} を {start_sec}s から再生 (フェードイン {fade_ms}ms)"
        )
    except Exception as e:
        print(f"❌ BGM再生エラー ({path}): {e}")


# トグル関数もフェードイン付きに
def toggle_bgm(fade_ms: int = 1000):
    global current_bgm
    current_bgm = "alert" if current_bgm == "normal" else "normal"
    play_bgm(current_bgm, fade_ms)


# 起動時に通常BGMを流す
play_bgm(current_bgm)

# ── 1) モード選択 ──────────────────
while True:
    mode = input("監修モード=1／非監修モード=0 を入力してください > ").strip()
    if mode in ("0", "1"):
        supervision_mode = mode == "1"
        break
    print("無効な入力です。0 か 1 を入力してください。")

# 統計ファイルのパス（同ディレクトリに置く）
stats_file = os.path.join(os.path.dirname(__file__), "supervision_stats.csv")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 姿勢推定のためのdetectorオブジェクトを作成
pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("✅ Mediapipe・モデル準備 完了")
if supervision_mode:
    # 監修モード：サイクルごとのインパルスを貯める
    impulse_records = {k: [] for k in part_keys}


if not os.path.exists(stats_file):
    raise FileNotFoundError(f"統計ファイルが見つかりません: {stats_file}")
df_stat = pd.read_csv(stats_file)
stats = {row["part"]: (row["mean"], row["std"]) for _, row in df_stat.iterrows()}
# ▶▶ 追加：Gauge UI の初期化（非監修モード時のみ）
if not supervision_mode:
    config_path = os.path.join(os.path.dirname(__file__), "positions.json")
    with open(config_path, "r", encoding="utf-8") as f:
        ui_conf = json.load(f)

    # GaugeDisplay の生成
    gauge = GaugeDisplay(config_path, stats, image_path="wheelchair_user.png")
    plt.ion()
    gauge.fig.show()
    # 非監修モードで更新する最新インパルス格納用
    current_impulses = {k: 0.0 for k in gauge.part_keys}
# ESC長押し検知用
ESC_HOLD_FRAMES = 10
esc_count = 0
# ── 2) 監修モード／非監修モードごとの準備 ──────────────────
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
"""lines = [
    ax.plot(np.linspace(-10, 0, 100), y, color=color, label=label)[0]
    for y, ax, color, label in zip(torque_sss, axes.flatten(), colors, labels)
]

for ax in axes.flatten():
    ax.set_xlim(-10, 0)  # X軸の範囲を-10から0まで設定
    # ax.set_ylim(0, 15000)  # X軸の範囲を-10から0まで設定

    ax.set_xlabel("時間(秒)")  # X軸ラベル
    ax.set_ylabel("トルク値")  # Y軸ラベル
    ax.legend()  # 凡例を表示"""

# plt.show()

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

print("Starting loop")
winsound.Beep(500, 1000)  # 周波数と持続時間を指定して音を鳴らす（例えば、500Hzで1秒間）

# 姿勢推定のための初期フレーム取得（再実行してOK）
ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

# Mediapipe用にRGB化 & 推定
frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
results0 = pose0.process(frame0_rgb)
results1 = pose1.process(frame1_rgb)

# キーポイント抽出（描画OFFでもOK）
frame0_kpts, frame1_kpts = extract_keypoints(
    results0, results1, pose_keypoints, frame0, frame1
)


# 有効なX座標のみ抽出
def get_valid_x_range(kpts):
    valid = [x for x, y in kpts if x >= 0]
    if not valid:
        return 0, frame0.shape[1]  # 検出なしならフル幅
    return min(valid), max(valid)


x0_min, x0_max = get_valid_x_range(frame0_kpts)
x1_min, x1_max = get_valid_x_range(frame1_kpts)

x_min = min(x0_min, x1_min)
x_max = max(x0_max, x1_max)
x_margin = 50

x_start = max(0, x_min - x_margin)
x_end = min(frame0.shape[1], x_max + x_margin)
print(f"✅ トリミング範囲: x = {x_start} ～ {x_end}")


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

    # トリミング処理の置き換え
    frame0 = frame0[:, x_start:x_end]
    frame1 = frame1[:, x_start:x_end]

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
    if WHILE_COUNT < 15 and WHILE_COUNT > 4:
        z_value += (transformed_p3ds[0][2]) / 10
    elif WHILE_COUNT == 15:
        detector = PushCycleDetector(z_value)
    else:
        pass

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
    MsL = [M1L, M2L, M3L, M4, M5L]
    FsL = [F1L, F2L, F3L, F4, F5L]
    partsL = [Part1L, Part2L, Part3L, Part4, Part5L]

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
    torquesL = calculate_individual_torques(
        MsL, FsL, np.array(r_g), tau_E, f_E, r_x, partsL, storage
    )
    link_upperarm_L = transformed_p3ds[3] - transformed_p3ds[1]
    link_forearm_L = transformed_p3ds[5] - transformed_p3ds[3]
    link_upperarm_R = transformed_p3ds[2] - transformed_p3ds[0]
    link_forearm_R = transformed_p3ds[4] - transformed_p3ds[2]
    link_shoulder_L = transformed_p3ds[1] - transformed_p3ds[0]
    link_shoulder_R = -link_shoulder_L
    global_wrist_R = torques[0][0]
    global_elbow_R = torques[1][0]
    global_shoulder_R = torques[2][0]
    local_wrist_R = compute_local_torque(global_wrist_R, link_forearm_R)
    local_elbow_R = compute_local_torque(global_elbow_R, link_upperarm_R)
    local_shoulder_R = compute_local_torque(global_shoulder_R, link_shoulder_R)
    global_wrist_L = torquesL[0][0]
    global_elbow_L = torquesL[1][0]
    global_shoulder_L = torquesL[2][0]
    local_wrist_L = compute_local_torque(global_wrist_L, link_forearm_L)
    local_elbow_L = compute_local_torque(global_elbow_L, link_upperarm_L)
    local_shoulder_L = compute_local_torque(global_shoulder_L, link_shoulder_L)

    # ディクショナリにトルク値を格納
    storage.add_torque("wrist_R", local_wrist_R)
    storage.add_torque("elbow_R", local_elbow_R)
    storage.add_torque("shoulder_R", local_shoulder_R)
    storage.add_torque("wrist_L", local_wrist_L)
    storage.add_torque("elbow_L", local_elbow_L)
    storage.add_torque("shoulder_L", local_shoulder_L)
    # torque_sssへのトルク値の追加（左右腕 6 要素）
    temp_local = [
        local_wrist_R,
        local_elbow_R,
        local_shoulder_R,
        local_wrist_L,
        local_elbow_L,
        local_shoulder_L,
    ]
    if WHILE_COUNT > 15:
        if detector.update(transformed_p3ds[0][2], WHILE_COUNT):
            cycle_switch = 1
            hist_len = len(current_torque_history[part_keys[0]])
            print("detected")
            if prev_cycle_frame is not None and hist_len >= min_history_len:
                if gauge is not None:
                    # GaugeDisplay に合わせた部位キーで最新インパルスを計算
                    for pk in gauge.part_keys:
                        # current_torque_history[pk] はリストとして保持
                        series = pd.Series(current_torque_history[pk])
                        pos = series[series > 0].sum() * dt
                        neg = series[series < 0].sum() * dt
                        imp = max(abs(pos), abs(neg))
                        impulse_records[pk].append(imp)
                        current_impulses[pk] = imp

                    # ゲージモジュールへ渡して描画更新
                    gauge.update_impulses(current_impulses)
                    gauge.update()
            # 次サイクル準備
            prev_cycle_frame = WHILE_COUNT
            for key in part_keys:
                current_torque_history[key].clear()

            # （任意）検出ログ出力
            print(f"Cycle impulse appended at frame {WHILE_COUNT}")
        else:
            cycle_switch = 0
        for key, vec in zip(part_keys, temp_local):
            current_torque_history[key].append(vec[2])
    else:
        pass
    # ノルムだけ取り出してプロット用に
    temp_norms = [np.linalg.norm(v) for v in temp_local]
    aim_torque.append(temp_local)

    # 新しい画像サイズを計算（余白分を加える）
    new_height = frame0.shape[0]
    new_width = frame0.shape[1] + PADDING

    # 描画フレーム生成
    new_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_frame[: frame0.shape[0], : frame0.shape[1]] = frame0

    # 余白部分に左右それぞれの局所トルクを表示
    labels = ["右手首", "右肘", "右肩", "左手首", "左肘", "左肩"]
    for i, (lbl, key) in enumerate(zip(labels, part_keys)):
        y = 40 + 30 * i
        # 今フレームのトルクノルム
        local_val = temp_norms[i]
        # 直前サイクルのインパルス（なければ 0 を表示）
        last_imp = impulse_records[key][-1] if impulse_records[key] else 0.0
        # テキスト形式を「T:トルク / I:インパルス」に
        text = f"{lbl} I:{last_imp:.1f}"
        new_frame = put_text_jp(
            new_frame,
            text,
            (new_width - 350, y),
            24,
            (255, 255, 255),
            20,
        )

    # グラフ用には10倍して丸め
    plot_vals = [int(v * 10) for v in temp_norms[:4]]

    # update_graphs(tuple(plot_vals), lines,        axes,        torque_sss,)
    cv.resizeWindow("MyWindow", new_width, new_height)
    cv.imshow("MyWindow", new_frame)
    # cv.imshow("cam1", frame1)
    WHILE_COUNT += 1
    end_time = time.perf_counter()
    dt = end_time - start_time
    print("dt=", f"{dt:.3f}")
    print("FPS:", 1 / dt)
    # ESC 長押しでモード終了
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        esc_count += 1
    else:
        esc_count = 0
    if esc_count > ESC_HOLD_FRAMES:
        print("← ESC長押し検出：ループを抜けます")
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
print(impulse_records)
# ── 4) ループ後処理 ──────────────────
if supervision_mode:
    # 監修モード終了：平均・標準偏差を計算し CSV 保存
    import statistics

    stats_out = []
    for key in part_keys:
        arr = impulse_records[key]
        if not arr:
            mu = std = 0.0
        else:
            mu = statistics.mean(arr)
            std = statistics.pstdev(arr)
        stats_out.append((key, mu, std))

    with open(stats_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "mean", "std"])
        writer.writerows(stats_out)
    print(f"監修モード統計値を保存しました: {stats_file}")

else:
    print("非監修モード終了（記録済み統計値を利用）")
pygame.mixer.music.stop()
pygame.mixer.quit()
print("▶️ BGM を停止しました")

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
