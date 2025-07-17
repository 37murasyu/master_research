import os
import numpy as np
from datetime import datetime

# ファイルパス関連
absolutepath = os.path.abspath(__file__)
folder_path = os.path.dirname(absolutepath)

# 時間刻みと体重（仮）
dt = 0.3  # 0.1秒ごと
w = 60  # 体重60kg

# 質量（リンクの部位ごとの質量）
# https://note.com/ss_sports_lab/n/nd284cb2c3628
m1 = w * 0.0227  # 上腕
m2 = w * 0.016  # 前腕
m3 = w * 0.006  # 手
m4 = w * 0.11  # 太腿

# 重力加速度ベクトル
g = np.array([0, 0, -9.81])
PADDING = 400  # 余白として追加するピクセル数
# add here if you need more keypoints

pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
# this will load the sample videos if no camera ID is given
# input_stream1 = folder_path + "\\media\\output1.mp4"
# input_stream2 = folder_path + "\\media\\output2.mp4"
# input_stream1 = folder_path + "\\media\\cam000_test.mp4"
# input_stream2 = folder_path + "\\media\\cam111_test.mp4"
input_stream1 = 0
input_stream2 = 1
# CSVファイルの絶対パス
rm_path = folder_path + "\\rm_method.csv"
# カメラの解像度を720pに設定
frame_shape = [720, 1280]
fps = 30
timestamp = datetime.now().strftime("%m%d_%H%M%S")
# 保存フォルダ（必要に応じて変更）
save_dir = "output_data"
os.makedirs(save_dir, exist_ok=True)
# ウィンドウ名
win_main = "MainMonitor"
win_second = "SecondMonitor"

win_main_point = [0, 0, 1280, 720]  # メインモニターのウィンドウ位置とサイズ
win_second_point = [1200, -1080, 3120, 0]  # セカンドモニターのウィンドウ位置とサイズ
SKIP_FRAMES = 0
WHILE_COUNT = 0
z_value = 0
cycle_switch = 0

part_keys = ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"]
# 各サイクルごとのインパルス（絶対値が大きい方）を格納する辞書
impulse_records = {k: [] for k in part_keys}
# 現在のサイクル内で z 成分を蓄積するリスト
current_torque_history = {k: [] for k in part_keys}
# 前回サイクル検出時のフレーム番号
prev_cycle_frame = None
min_history_len = 3  # ガード用
detector = None  # 既に初期化済みと仮定
gauge = None
current_impulses = {}
