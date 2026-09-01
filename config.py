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

# MediaPipe Pose ランドマーク出力で使用するインデックス。
# bodypose3d の順序に合わせて12点のみ（右手首→右肘→右肩→左肩→左肘→左手首→右/左腰→右/左膝→右/左足首）。
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
# this will load the sample videos if no camera ID is given
# input_stream1 = folder_path + "\\media\\output1.mp4"
# input_stream2 = folder_path + "\\media\\output2.mp4"
# input_stream1 = folder_path + "\\media\\cam000_test.mp4"
# input_stream2 = folder_path + "\\media\\cam111_test.mp4"

# 入力ストリーム（デフォルトはカメラID 0/1）
input_stream1 = 0
input_stream2 = 1

# 環境変数で上書き可能にする（例）
#   PowerShell:
#     $env:CAM0 = "1"
#     $env:CAM1 = "video=HD Pro Webcam C920"   # MSMF/DSHOW 名指定
#     python .\master_research_code.py
def _parse_cam_env(val: str | None):
    if not val:
        return None
    v = val.strip()
    # 数字だけなら index、そうでなければ文字列のまま（"video=..." 等）
    if v.lstrip("-+").isdigit():
        try:
            return int(v)
        except ValueError:
            return v
    return v

_env_cam0 = _parse_cam_env(os.environ.get("CAM0"))
_env_cam1 = _parse_cam_env(os.environ.get("CAM1"))
if _env_cam0 is not None:
    input_stream1 = _env_cam0
if _env_cam1 is not None:
    input_stream2 = _env_cam1

# 追加オプション: サンプル動画強制/自動フォールバック
# USE_SAMPLE_VIDEOS=1 なら常に動画ファイルを使用（下記の自動検出: 最新の cam0_output_*.mp4 / cam1_output_*.mp4 を優先、なければ media/cam000_test.mp4 等）
# AUTO_FALLBACK_TO_FILES=1 なら、カメラが開けなかった場合に自動で動画ファイルへ切替
USE_SAMPLE_VIDEOS = int(os.environ.get("USE_SAMPLE_VIDEOS", "0"))
AUTO_FALLBACK_TO_FILES = int(os.environ.get("AUTO_FALLBACK_TO_FILES", "1"))

# 完全ヘッドレス実行（OpenCV/Matplotlib のウィンドウや waitKey を使わない）
# HEADLESS=1 にするとデバッグモードでのクラッシュ回避に有効です
HEADLESS = int(os.environ.get("HEADLESS", "0"))

# I/O 詳細ログの出力制御
IO_DEBUG = int(os.environ.get("IO_DEBUG", "0"))

# カメラが開けない場合のフォールバック優先度
# 0: サンプル動画 (media\cam000_test.mp4 / cam111_test.mp4) を優先（推奨・デフォルト）
# 1: 最新の録画ペア (cam0/1_output_*.mp4) を優先
PREFER_RECORDING_PAIRS = int(os.environ.get("PREFER_RECORDING_PAIRS", "1"))
# CSVファイルの絶対パス
rm_path = folder_path + "\\rm_method.csv"
# カメラの解像度を720pに設定
frame_shape = [720, 1280]
fps = 30
# 実行毎に新しいタイムスタンプを生成。バッチ処理等で固定したい場合は環境変数 TIMESTAMP_OVERRIDE を設定。
_ts_override = os.environ.get("TIMESTAMP_OVERRIDE", "").strip()
if _ts_override:
    # 簡易バリデーション (MMDD_HHMMSS 形式を想定、数字と '_' のみ許可)
    import re as _re
    if _re.fullmatch(r"[0-1][0-9][0-3][0-9]_[0-2][0-9][0-5][0-9][0-5][0-9]", _ts_override):
        timestamp = _ts_override
    else:
        # フォーマット不一致なら通常生成にフォールバック
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
else:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
# 保存フォルダ（必要に応じて変更）
save_dir = "output_data"
os.makedirs(save_dir, exist_ok=True)
# ウィンドウ名
win_main = "MainMonitor"
win_second = "SecondMonitor"

win_main_point = [0, 0, 1280, 720]  # メインモニターのウィンドウ位置とサイズ
win_second_point = [1200, -1080, 3120, 0]  # セカンドモニターのウィンドウ位置とサイズ
SKIP_FRAMES = int(os.environ.get("SKIP_FRAMES", "0"))
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
