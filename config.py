import os
import numpy as np
from datetime import datetime

# ファイルパス関連
absolutepath = os.path.abspath(__file__)
folder_path = os.path.dirname(absolutepath)

# 体重（仮）
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

# MediaPipe Pose ランドマーク出力で使用するインデックス（12 点）。
#
# **このリストの並び順に意味は無い。** 元実装 TemugeB/bodypose3d は
#     for i, landmark in enumerate(results.pose_landmarks.landmark):
#         if i not in pose_keypoints: continue
# とランドマーク ID の昇順に走査しており、pose_keypoints は「どの点を使うか」の
# フィルタとしてしか働かない。したがって 3D 点列の並びは常に昇順になる:
#
#   [0]=11 左肩  [1]=12 右肩  [2]=13 左肘  [3]=14 右肘  [4]=15 左手首  [5]=16 右手首
#   [6]=23 左腰  [7]=24 右腰  [8]=25 左膝  [9]=26 右膝  [10]=27 左足首 [11]=28 右足首
#
# master_research_code.py の part_calculations や慣性テンソルのリンク長は
# この並びを前提にしている。抽出側は sorted() を通すこと（再検算 R-1）。
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
rm_path = os.path.join(folder_path, "rm_method.csv")
# カメラの解像度を720pに設定
frame_shape = [720, 1280]
fps = 30

# 力学計算のサンプル間隔のフォールバック。
#
# 以前は 10 行目に dt = 0.3（コメントは「0.1秒ごと」）と書かれており、値もコメントも
# fps も三者三様に食い違っていた。微分と積分が必要とするのは「連続して処理される
# フレームの実時間間隔」であって、それは間引き設定に依存するため定数では決まらない。
# master_research_code.py は起動時に間引き係数から _DYN_DT を算出して使う。
# ここに残しているのは twin_video_capture.py と master_research_code_00.py 向けの後方互換。
dt = 1.0 / fps


def resolve_dynamics_dt(src_fps, *, fixed_hz_on, fixed_skip, skip_mod=1, override=None):
    """力学計算に使うサンプル間隔 [秒] を決める。

    速度・加速度・角速度の微分と、エネルギー・力積の積分が必要とするのは
    **連続して処理されるフレームの実時間間隔**であって、カメラのフレーム間隔ではない。
    実行時は既定でフレームを間引く（RT_POSE_FIXED_HZ_ON / SKIP_FRAMES）ため、
    1/fps では間隔を過小に見積もる。

    Parameters
    ----------
    src_fps:
        カメラが実際に出しているフレームレート。
    fixed_hz_on:
        固定 Hz 間引き（RT_POSE_FIXED_HZ_ON）が有効か。
    fixed_skip:
        固定 Hz 間引きで飛ばすフレーム数（処理間隔は fixed_skip + 1 フレーム）。
    skip_mod:
        SKIP_FRAMES による間引き。1 なら間引きなし。
    override:
        文字列または数値。与えられればそれをそのまま採る（DT_SEC 用）。

    戻り値は (dt_sec, 由来を説明する文字列)。
    """
    if override not in (None, ""):
        value = float(override)
        if not (value > 0):
            raise ValueError(f"dt は正の値である必要があります: {override!r}")
        return value, f"override={override}"

    stride = (int(fixed_skip) + 1) if fixed_hz_on else max(1, int(skip_mod))
    rate = float(src_fps)
    if not (rate > 0):
        rate = float(fps)
    return stride / rate, f"{stride}frame / {rate:.3f}fps"
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


# ===================== 体節の重心比とリンク定義 =====================
#
# 重心比は「近位端からの距離 / リンク長」。Winter, *Biomechanics and Motor Control
# of Human Movement* の体節パラメータ表に基づく。リポジトリ内に同じ値が
# compute_torque_from_pose.py・compute_cycle_energy_elbow_wrist.py・
# inverse_dynamics_two_link.py・compute_elbow_cycle_work.py と散在していたのを
# ここに集約した（再検算 H-9）。
COM_FRACTIONS = {
    "upper_arm": 0.436,
    "forearm": 0.430,
    "hand": 0.506,
    "thigh": 0.433,   # 出典要確認。上 3 つと違い、リポジトリ内に既存値が無かった
}

# リンク定義。索引は pose_keypoints をランドマーク ID の昇順に並べたときの位置
# （[0]左肩 [1]右肩 [2]左肘 [3]右肘 [4]左手首 [5]右手首 [6]左腰 [7]右腰
#   [8]左膝 [9]右膝 [10]左足首 [11]右足首）。
#
# start/end は **遠位 → 近位** の向きで書かれている（例: upper_arm_R は肘→肩）。
# したがって重心は end 側（近位端）から測る:
#     centroid = p_end + com_fraction * (p_start - p_end)
# com_fraction = 0.5 なら両端の中点になり、2026-09-08 以前の挙動と一致する。
#
# both_shoulder / both_hip は体節ではなく「両肩の中点」「両腰の中点」であり、
# r_g の組み立て側（master_research_code.py の r_g_R）が肩:腰 = 3:1 の重み付けで
# 上胴体・下胴体の重心を作る。したがってここは 0.5 のままにする。
part_calculations = {
    "upper_arm_R": {"start": 3, "end": 1, "com_fraction": COM_FRACTIONS["upper_arm"]},
    "forearm_R": {"start": 5, "end": 3, "com_fraction": COM_FRACTIONS["forearm"]},
    "both_shoulder": {"start": 0, "end": 1, "com_fraction": 0.5},
    "both_hip": {"start": 6, "end": 7, "com_fraction": 0.5},
    "up_arm_l": {"start": 2, "end": 0, "com_fraction": COM_FRACTIONS["upper_arm"]},
    "forearm_L": {"start": 4, "end": 2, "com_fraction": COM_FRACTIONS["forearm"]},
    "upper_Leg_R": {"start": 7, "end": 9, "com_fraction": COM_FRACTIONS["thigh"]},
    "upper_Leg_L": {"start": 6, "end": 8, "com_fraction": COM_FRACTIONS["thigh"]},
}

# 慣性テンソルのリンク長を決めるのに使うフレーム数。
# 1 フレームの瞬時値だと三角測量の誤差がそのまま全実行に固定される。
# 慣性回帰式 I = a*w + b*l + c は l に極端に敏感なので中央値で均す（再検算 R-6）。
INERTIA_LENGTH_FRAMES = int(os.environ.get("INERTIA_LENGTH_FRAMES", "30"))
