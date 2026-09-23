import os
import numpy as np
from datetime import datetime

# ファイルパス関連
absolutepath = os.path.abspath(__file__)
# 凍結アプリではこのファイルの隣はバンドル内（読み取り専用）なので、ワーカーが用意した
# ワークスペースを基点にする（app/core/workspace.py）。開発時は未設定でリポジトリルート。
folder_path = os.environ.get("APP_WORKSPACE") or os.path.dirname(absolutepath)

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

# MediaPipe Pose ランドマーク出力で使用するインデックス（16 点）。
#
# **このリストの並び順に意味は無い。** 元実装 TemugeB/bodypose3d は
#     for i, landmark in enumerate(results.pose_landmarks.landmark):
#         if i not in pose_keypoints: continue
# とランドマーク ID の昇順に走査しており、pose_keypoints は「どの点を使うか」の
# フィルタとしてしか働かない。したがって 3D 点列の並びは常に昇順になる:
#
#   [0]=11 左肩   [1]=12 右肩   [2]=13 左肘   [3]=14 右肘
#   [4]=15 左手首 [5]=16 右手首 [6]=17 左小指 [7]=18 右小指
#   [8]=19 左人差指 [9]=20 右人差指 [10]=23 左腰 [11]=24 右腰
#   [12]=25 左膝 [13]=26 右膝  [14]=27 左足首 [15]=28 右足首
#
# **位置索引を直書きしないこと。** リンク定義は PART_LINK_IDS（下）に
# ランドマーク名で書き、build_part_calculations() が索引を組み立てる。
# 抽出側は sorted() を通すこと（再検算 R-1）。
#
# 手のランドマーク（17〜20）を含めるのは、手首の関節軸を手のひら面から
# 決めるため。従来は「リンク＝前腕、親＝上腕」で軸を作っており、
# 手首の掌屈/背屈軸ではなく肘の屈曲軸を見ていた（グローバルトルク 149 N·m のうち
# local_y が 6.4 N·m しか拾えていなかった）。
#
# 計算コストは問題にならない。pose_landmarker_lite.task を num_poses=1 で
# 走らせており MediaPipe は常に 33 点すべてを推論する。ここは抽出フィルタでしかない。
# 増えるのは三角測量だけで、12 → 16 点で +0.016 ms（30fps 予算の +0.05%）。
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 20, 18, 19, 17]
# this will load the sample videos if no camera ID is given
# input_stream1 = folder_path + "\\media\\output1.mp4"
# input_stream2 = folder_path + "\\media\\output2.mp4"
# input_stream1 = folder_path + "\\media\\cam000_test.mp4"
# input_stream2 = folder_path + "\\media\\cam111_test.mp4"

_ENV_TRUE = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "false", "no", "off")


def env_flag(name, default):
    """環境変数を真偽値として読む。大文字小文字と前後の空白は問わない。

    1 / true / yes / on なら True、0 / false / no / off なら False、未設定やそれ以外は default。
    かつて master_research_code.py は ``in ('1','true','True')`` と ``not in ('0','false','False')`` の
    2 通りで読んでおり、''・'yes'・'TRUE' などで結果が逆になっていた（KNOWN_ISSUES §4-4）。
    設定スキーマの抽出器（tools/extract_env_schema.py）はこの呼び出しを bool として拾う。
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return bool(default)


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
# 保存フォルダ。GUI は「出力先」と表示している場所（app.core.settings.measurement_output_dir）を環境変数
# OUTPUT_DIR で渡す。GUI を通さずに起動したときは、従来どおり作業フォルダの output_data。
# 名前を定数に入れるのは、設定画面の項目（tools/extract_env_schema.py が拾う）にしないため
_OUTPUT_DIR_ENV = "OUTPUT_DIR"
save_dir = os.environ.get(_OUTPUT_DIR_ENV, "").strip() or "output_data"
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
    "hand": 0.506,    # 手首から中指の中手骨頭まで。理論 1RM 仕事量の分母で使う
    "thigh": 0.433,   # Winter の表の大腿（近位 0.433 / 遠位 0.567）。2026-09-23 に照合
}

# 体節の質量比（体重に対する比）。Winter の体節パラメータ表。
#
# ただし上腕の 0.0227 は Winter（0.028）ではなく、config.m1 と慣性回帰式の行（utils_dynamic の
# _SEGMENT_MASS_FRACTION）に合わせた値。出典は未確認（KNOWN_ISSUES §5-5）。
# 初期コミットの参考URLのサンプルコードは 0.027 であり、0.0227 の根拠にはならない。
#
# 手の慣性テンソル（慣性係数 CSV の行 5）はどこでも使わない。プッシュアップの鎖では手を
# アームレストに置いた固定端とし、腕を肩から吊る鎖では手首の質点とする（push_up_model）。
# 手の慣性は m L²/12 ≈ 0.36 × 0.08² / 12 ≈ 2×10⁻⁴ kg·m² で、前腕の 1/5 程度しかない。
#
# リポジトリ内には別系統の値もある（utils_dynamic の上胴体 0.276 / 下胴体 0.19、
# 合わせて体幹 0.466）。そちらは utils_dynamic.calculate_M_and_F の肩の質量配分
# でも使われているが出典が不明で、全身合計が 0.978 と 1.0 から離れる。
# Winter 系は合計 0.989 でより近いのでこちらを採る（保留 H-A。差は体幹＋頭で 5.7%）。
SEGMENT_MASS_FRACTIONS = {
    "trunk": 0.497,
    "head_neck": 0.081,
    "upper_arm": 0.0227,
    "forearm": 0.016,
    "hand": 0.006,
    "thigh": 0.100,
    "shank": 0.0465,
    "foot": 0.0145,
}

# 座位プッシュアップで**腕が持ち上げる**質量の比。
#
# モデル: 手を外力との境界（アームレストに固定された端）とし、そこから近位
# ── 前腕 → 上腕 → 肩 ── の鎖を解く。肩より近位（体幹と頭）は剛体として扱わず、
# 「肩に載る外力」に集約する。脚は床／フットレストが支えるので含めない。
#
# 前腕・上腕は鎖の一部として自重が入るので、ここには含めない。
# 手はアームレストに接しており持ち上げられないので、やはり含めない。
SUPPORTED_MASS_FRACTION = (
    SEGMENT_MASS_FRACTIONS["trunk"] + SEGMENT_MASS_FRACTIONS["head_neck"]
)

# 片腕が支える割合。両手で対称に押す想定なので 0.5。
SUPPORT_SHARE_DEFAULT = 0.5

# MediaPipe Pose のランドマーク ID に名前を付ける。
MP_LANDMARK = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_PINKY": 17, "R_PINKY": 18,
    "L_INDEX": 19, "R_INDEX": 20,
    "L_THUMB": 21, "R_THUMB": 22,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
}


def landmark_slots(keypoints=None):
    """ランドマーク ID → 3D 点列における位置索引 の対応を作る。

    抽出は ID の昇順で行われる（utils.extract_keypoints）ので、
    並びは sorted(keypoints) で決まる。
    """
    return {pid: i for i, pid in enumerate(sorted(
        pose_keypoints if keypoints is None else keypoints))}


SLOT = landmark_slots()


def slot_of(name, slots=None):
    """ランドマーク名（MP_LANDMARK のキー）から位置索引を引く。

    pose_keypoints に含まれていない点を指すと KeyError で落ちる。
    黙って別の関節を指すよりよい（再検算 R-1 の再発防止）。
    """
    return (SLOT if slots is None else slots)[MP_LANDMARK[name]]


# リンク定義の正本。**位置索引ではなくランドマーク名で書く。**
#
# 位置索引を直書きすると、pose_keypoints に点を足したときに昇順の並びが変わり、
# 索引が別の関節を指すようになる。例えば手のランドマーク（17〜20）を足すと
# 腰・膝・足首が [6..11] から [10..15] へずれ、both_hip が「両小指」を指す。
# 上肢（[0..5]）は偶然無傷なので、テストが上肢しか見ていないと気づけない。
#
# start/end は **遠位 → 近位** の向きで書く（例: upper_arm_R は肘→肩）。
# したがって重心は end 側（近位端）から測る:
#     centroid = p_end + com_fraction * (p_start - p_end)
# com_fraction = 0.5 なら両端の中点になり、2026-09-08 以前の挙動と一致する。
#
# both_shoulder / both_hip は体節ではなく「両肩の中点」「両腰の中点」であり、
# r_g の組み立て側（master_research_code.py の r_g_R）が肩:腰 = 3:1 の重み付けで
# 上胴体・下胴体の重心を作る。したがってここは 0.5 のままにする。
PART_LINK_IDS = {
    "upper_arm_R": ("R_ELBOW", "R_SHOULDER", COM_FRACTIONS["upper_arm"]),
    "forearm_R": ("R_WRIST", "R_ELBOW", COM_FRACTIONS["forearm"]),
    "both_shoulder": ("L_SHOULDER", "R_SHOULDER", 0.5),
    "both_hip": ("L_HIP", "R_HIP", 0.5),
    "up_arm_l": ("L_ELBOW", "L_SHOULDER", COM_FRACTIONS["upper_arm"]),
    "forearm_L": ("L_WRIST", "L_ELBOW", COM_FRACTIONS["forearm"]),
    "upper_Leg_R": ("R_HIP", "R_KNEE", COM_FRACTIONS["thigh"]),
    "upper_Leg_L": ("L_HIP", "L_KNEE", COM_FRACTIONS["thigh"]),
}


def build_part_calculations(keypoints=None):
    """PART_LINK_IDS から位置索引つきのリンク定義を組み立てる。

    keypoints を渡せば任意の構成で索引を計算できる（テスト用）。
    """
    slots = landmark_slots(keypoints)
    return {
        name: {"start": slot_of(start, slots), "end": slot_of(end, slots),
               "com_fraction": com}
        for name, (start, end, com) in PART_LINK_IDS.items()
    }


part_calculations = build_part_calculations()

# 出力ファイルの規約の版（KNOWN_ISSUES §5-6）。トルクや仕事率の意味を変えたら上げる。
# 版はオフラインの *_meta.json と、リアルタイム経路の出力ファイル名（_s{版} を付ける）に入れる。
# 既存のファイルは書き換えない。版の無いファイルは v1。
#
#   v1（〜2026-09-22）: 版の記録なし。2026-09-08 に左右ラベルの入れ替え（再検算 R-1）を含む
#   v2（2026-09-23）: 共有モデル push_up_model に一本化
#       - 重力を初期フレームの体幹の向きから決める（§1-5。オフラインは奥行き方向を向いていた）
#       - wrist_* は手首まわり、elbow_* は肘まわり（§5-7。USB・スマホ経路は 1 つずれていた）
#       - 手を固定端に、体幹＋頭の荷重を肩に載せる鎖（§2-1）を 3 経路とも使う（§5-8）
#       - 手首の局所軸は手のひら（手の点）か肘の屈曲軸（§5-1）
#       - トルクの向き: 手首は「手が前腕に」、肘は「前腕が上腕に」加えるトルク（手を固定端とする鎖の
#         反力側）。肩は「体幹が吊った腕に」加えるトルク。局所 y = 親 × リンク（push_up_model.joint_axes）。
#         このため手首・肘の τ_y の符号は解剖学の慣例（近位が遠位に加える）と逆で、v1 の既定
#         （--wrist-base なし）の肘とも逆になる。仕事率 τ·ω_rel は向きの取り方に依らない
#       - 慣性テンソルをリンクの向きに合わせて回す（§2-3）
#       - オフラインのトルク倍率の既定を 0.01 から 1 に（m 入力で 1/100 になっていた）
OUTPUT_SCHEMA_VERSION = 2

# 慣性テンソルのリンク長を決めるのに使うフレーム数。
# 1 フレームの瞬時値だと三角測量の誤差がそのまま全実行に固定される。
# 慣性回帰式 I = a*w + b*l + c は l に極端に敏感なので中央値で均す（再検算 R-6）。
INERTIA_LENGTH_FRAMES = int(os.environ.get("INERTIA_LENGTH_FRAMES", "30"))


# ===================== 理論仕事量の係数 =====================
#
# 1 サイクルの角度範囲にわたる cos θ の積分。∫_a^b cos θ dθ = sin(b) - sin(a)。
# 値ではなく角度範囲で持つのは、論文の定義が変わったときに 1 箇所で追随できるようにするため。
#
#   -90°〜45° → sin(45°) - sin(-90°) = √2/2 + 1 = 1.7071
#   -90°〜60° → sin(60°) - sin(-90°) = √3/2 + 1 = 1.8660
#
# 2026-09-08 以前は 2 系統が併存していた（再検算 R-5）:
#   - master_research_code.py と offline_wrist_energy.py が √3/2 + 1 = 1.8660
#   - compute_cycle_energy_elbow_wrist.py ほか 3 箇所が 16.73 = (√2/2 + 1) × 9.8
# 9.3% 食い違ったまま二重管理されていた。
#
# **-90°〜45° に統一した。** 根拠は 力学計算_検証結果.md の C 節が 16.73 を
# 「導出も値も正しい」と検算しており、論文のスコアもこの系統で算出されているため。
# 論文本文の定義が -90°〜60° だと判明した場合は下の 1 行だけを直すこと。
WORK_ANGLE_RANGE_DEG = (-90.0, 45.0)
WORK_INTEGRAL_K = float(
    np.sin(np.radians(WORK_ANGLE_RANGE_DEG[1])) - np.sin(np.radians(WORK_ANGLE_RANGE_DEG[0]))
)

# 重力加速度の大きさ。g（ベクトル）と食い違わせないためここから引く。
# かつては config.g=9.81 / offline_wrist_energy.G=9.80665 / 16.73 に内包された 9.8 の
# 3 つが併存していた。
G_SCALAR = float(np.linalg.norm(g))

# 理論仕事量 W = (m_x·r_g + m_max·r_x) × THEORETICAL_WORK_COEFF。
# 従来 16.73 と直書きされていた値に相当する（16.73 は g=9.8 を内包していたので
# G_SCALAR=9.81 に揃えた分だけ 0.1% 増える）。
THEORETICAL_WORK_COEFF = WORK_INTEGRAL_K * G_SCALAR

# 部位別の等価質量係数（体重に対する比）。
# wrist: 上腕 0.026 + 体幹（上胴体＋下胴体、0.276 + 0.19） + 太もも 0.123
# elbow: 体幹（上胴体＋下胴体、0.276 + 0.19） + 太もも 0.123
# master_research_code.py が和を組み立て、offline_wrist_energy.py が潰した値を
# 直書きしており、片方だけ直すと食い違う状態だった。
EFFECTIVE_MASS_COEFFS = {
    "upper_arm": 0.026,
    "torso": 0.276 + 0.19,
    "thigh": 0.123,
}
EFFECTIVE_MASS_BY_JOINT = {
    "wrist": (EFFECTIVE_MASS_COEFFS["upper_arm"]
              + EFFECTIVE_MASS_COEFFS["torso"]
              + EFFECTIVE_MASS_COEFFS["thigh"]),
    "elbow": EFFECTIVE_MASS_COEFFS["torso"] + EFFECTIVE_MASS_COEFFS["thigh"],
    "shoulder": 0.0,   # 仕様未定
}
