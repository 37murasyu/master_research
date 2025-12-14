import textwrap
import os
import time

# pylint: disable=no-member
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import folder_path


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P


# Triangulation using OpenCV (replaces SVD-based DLT)
def DLT(P1, P2, point1, point2):
    """
    2視点の射影行列 P1, P2 とピクセル座標 point1, point2 から、
    OpenCV の cv.triangulatePoints を用いて 3D 点を推定する。

    Parameters
    ----------
    P1 : np.ndarray (3x4)
    P2 : np.ndarray (3x4)
    point1 : Sequence[float] -> [x,y]
    point2 : Sequence[float] -> [x,y]

    Returns
    -------
    np.ndarray shape (3,)
        三次元点 (x,y,z)。w が不正な場合は [-1,-1,-1]。
    """
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    # 2x1 の配列にして渡す
    pts1 = np.array([[float(point1[0])], [float(point1[1])]], dtype=np.float64)
    pts2 = np.array([[float(point2[0])], [float(point2[1])]], dtype=np.float64)

    Xh = cv.triangulatePoints(P1, P2, pts1, pts2)  # 4xN (ここでは N=1)
    w = float(Xh[3, 0])
    if not np.isfinite(w) or abs(w) < 1e-12:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    X = (Xh[:3, 0] / w).astype(np.float64)
    if not np.all(np.isfinite(X)):
        return np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    return X


def read_camera_parameters(camera_id, savefolder=folder_path + "\\camera_parameters\\"):
    path = f"{savefolder}c{camera_id}.dat"
    with open(path, "r", encoding="utf-8") as inf:
        _ = inf.readline()
        cmtx = [[float(en) for en in inf.readline().split()] for _ in range(3)]
        _ = inf.readline()
        dist = [float(en) for en in inf.readline().split()]
    return np.array(cmtx), np.array([dist])


def read_rotation_translation(camera_id, savefolder=folder_path + "\\camera_parameters\\"):
    path = f"{savefolder}rot_trans_c{camera_id}.dat"
    with open(path, "r", encoding="utf-8") as inf:
        _ = inf.readline()
        rot = [[float(en) for en in inf.readline().split()] for _ in range(3)]
        _ = inf.readline()
        trans = [[float(en) for en in inf.readline().split()] for _ in range(3)]
    return np.array(rot), np.array(trans)


def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis=1)
    else:
        return np.concatenate([pts, [1]], axis=0)


def get_projection_matrix(camera_id, file_mode):
    base = folder_path + ("\\camera_parameters\\Param_for_MYvideo\\" if file_mode else "\\camera_parameters\\")
    cmtx, _ = read_camera_parameters(camera_id, base)
    rvec, tvec = read_rotation_translation(camera_id, base)
    return cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3]


def write_keypoints_to_disk(filename, kpts):
    with open(filename, "w", encoding="utf-8") as fout:
        for frame_kpts in kpts:
            for kpt in frame_kpts:
                if len(kpt) == 2:
                    fout.write(str(kpt[0]) + " " + str(kpt[1]) + " ")
                else:
                    fout.write(str(kpt[0]) + " " + str(kpt[1]) + " " + str(kpt[2]) + " ")

            fout.write("\n")


def extract_keypoints(results0, results1, pose_keypoints, frame0, frame1):
    """
    MediaPipeの姿勢推定結果から、指定されたキーポイントのみを抽出し、
    ピクセル座標に変換して2つのフレーム（frame0とframe1）上に描画する関数。

    キーポイントが検出されなかった場合は、各キーポイント位置に [-1, -1] を設定します。

    Parameters:
        results0 (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            フレーム0に対応する姿勢推定結果（MediaPipeの出力）。
        results1 (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            フレーム1に対応する姿勢推定結果。
        pose_keypoints (List[int]):
            抽出対象となる関節インデックスのリスト。これに含まれるキーポイントだけを抽出・描画する。
        frame0 (np.ndarray):
            キーポイントを描画する対象となる画像（フレーム0）。
        frame1 (np.ndarray):
            キーポイントを描画する対象となる画像（フレーム1）。

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            - frame0_keypoints: 抽出されたキーポイントのピクセル座標リスト（frame0用）。
            - frame1_keypoints: 同上（frame1用）。
            各キーポイントは [x, y] の形式で、未検出時は [-1, -1]。
    """

    draw_kpts = os.getenv('DRAW_KEYPOINTS', '1') not in ('0','false','False')

    def _extract(results, frame):
        if not results.pose_landmarks:
            return [[-1, -1]] * len(pose_keypoints)
        out = []
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i not in pose_keypoints:
                continue
            px = int(round(landmark.x * frame.shape[1]))
            py = int(round(landmark.y * frame.shape[0]))
            if draw_kpts:
                cv.circle(frame, (px, py), 3, (0, 0, 255), -1)
                cv.putText(frame, str(i), (px + 5, py - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            out.append([px, py])
        return out or [[-1, -1]] * len(pose_keypoints)

    return _extract(results0, frame0), _extract(results1, frame1)


def calculate_3d_keypoints(frame0_keypoints, frame1_keypoints, P0, P1, _=None):
    """
    2視点からの2Dキーポイント情報を用いて、DLT法により3次元キーポイントを計算する関数。

    各キーポイントについて、どちらか一方でも未検出（[-1, -1]）であれば、
    対応する3D座標は [-1, -1, -1] として出力されます。

    Parameters:
        frame0_keypoints (List[List[int]]):
            フレーム0における2Dキーポイントのリスト。各点は [x, y] のピクセル座標。
        frame1_keypoints (List[List[int]]):
            フレーム1における2Dキーポイントのリスト。
        P0 (np.ndarray):
            フレーム0に対応するカメラの投影行列（3x4）。
        P1 (np.ndarray):
            フレーム1に対応するカメラの投影行列（3x4）。
        pose_keypoints (List[int]):
            対象となるキーポイントのインデックスリスト（この関数内では未使用だが、整合性保持のため引数に含まれている）。

    Returns:
        List[List[float]]:
            再構成された3次元キーポイントのリスト。各点は [x, y, z] の形式。
            未検出点は [-1, -1, -1] で表現される。
    """
    frame_p3ds = []
    for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        if uv1[0] == -1 or uv2[0] == -1:
            _p3d = [-1, -1, -1]

        else:
            _p3d = DLT(P0, P1, uv1, uv2)

        frame_p3ds.append(_p3d)
    return frame_p3ds


def put_text_jp(img, text, position, font_size, color, line_width):
    """
    OpenCV画像に日本語テキストを描画する関数。

    OpenCV形式の画像に対して、日本語フォントに対応したテキストを指定位置に描画し、
    改行幅を考慮して整形した後、新たな画像をOpenCV形式で返す。

    Parameters:
        img (numpy.ndarray): OpenCV形式の入力画像。
        text (str): 描画する日本語テキスト。
        position (tuple): テキストの描画位置（x, y）。
        font_size (int): フォントサイズ。
        color (tuple): テキストの色（R, G, B）。
        line_width (int): 1行あたりの最大文字数（改行幅）。

    Returns:
        numpy.ndarray: テキストが描画されたOpenCV形式の画像。
    """
    _t0 = time.perf_counter()
    img_pil = Image.fromarray(img)
    _t1 = time.perf_counter()

    # フォント取得（キャッシュあり）
    def _get_jp_font(sz: int):
        # シンプルなキャッシュ
        cache_key = f"meiryo_{sz}"
        font_obj = _FONT_CACHE.get(cache_key)
        if font_obj is not None:
            return font_obj
        try:
            font_path = folder_path + "\\meiryo\\meiryo.ttc"
            font_obj = ImageFont.truetype(font_path, sz)
        except OSError:
            font_obj = ImageFont.load_default()
        _FONT_CACHE[cache_key] = font_obj
        return font_obj

    draw = ImageDraw.Draw(img_pil)
    _t2a = time.perf_counter()
    font = _get_jp_font(int(font_size))
    _t2b = time.perf_counter()

    wrapped_text = textwrap.fill(text, width=line_width)
    _t3 = time.perf_counter()
    draw.text(position, wrapped_text, font=font, fill=color)
    _t4 = time.perf_counter()
    out = np.array(img_pil)
    _t5 = time.perf_counter()

    if os.getenv('PERF_DRAW_TRACE', '0') in ('1','true','True'):
        print(
            "[DRAW_TIMING] fromarray={:.2f}ms font={:.2f}ms wrap={:.2f}ms draw={:.2f}ms toarray={:.2f}ms".format(
                (_t1 - _t0) * 1000.0,
                (_t2b - _t2a) * 1000.0,
                (_t3 - _t2b) * 1000.0,
                (_t4 - _t3) * 1000.0,
                (_t5 - _t4) * 1000.0,
            )
        )
    return out

# モジュール内フォントキャッシュ（サイズ毎）
_FONT_CACHE = {}


def display_choices(question, a, _=None):
    """
    質問と2つの選択肢を表示し、ユーザーのキーボード入力に応じて選択を受け付けるGUI関数。

    OpenCVを用いて黒背景のウィンドウを表示し、指定された質問文と選択肢A/Bを表示する。
    矢印キーの代わりに 'u'（上）と 'd'（下）キーで選択肢を切り替え、Enterキーで決定する。
    また、'q'キーでキャンセル（強制終了）可能。

    Parameters
    ----------
    question : str
        表示する質問文（日本語対応）。
    a : str
        選択肢Aのテキスト。
    b : str
        選択肢Bのテキスト。

    Returns
    -------
    selection : int
        ユーザーが選んだ選択肢のインデックス（0: a, 1: B）。

    Notes
    -----
    - フォント表示には日本語対応の `put_text_jp` 関数を使用する必要があります。
    - OpenCVのGUI機能（`cv.imshow`, `cv.waitKey`）に依存しています。
    - 上下の選択は 'u'（上）と 'd'（下）で行うように指定されています。
    - 関数の末尾の `cv.destroyAllWindows()` は `return` の前に移動すべきです（現状では呼ばれません）。
    """

    font_size = 24
    color = (255, 255, 255)  # 白色
    selection = 0  # 選択肢のインデックス (0: a, 1: b)

    # 画像を作成（背景は黒）
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    while True:
        img_copy = img.copy()
        # テキストの表示
        img_copy = put_text_jp(img_copy, question, (10, 50), font_size, color, 20)
        img_copy = put_text_jp(img_copy, a, (50, 100), font_size, color, 20)
        # img_copy = put_text_jp(img_copy, b, (50, 150), font_size, color, 20)

        # 矢印の表示
        arrow_pos = (30, 100 + 50 * selection)  # 矢印の位置を選択肢に合わせて調整
        img_copy = put_text_jp(img_copy, "→", arrow_pos, font_size, color, 20)

        # 画像の表示
        cv.imshow("Choice", img_copy)

        # キーボード入力を待機
        key = cv.waitKey(0) & 0xFF
        # print(key)
        if key == ord("q"):  # 'q' で終了
            break
        elif (
            key == 117 and selection > 0
        ):  # Uキー (cv2.KEY_UP_ARROW のキーコードに置き換えてください)
            selection -= 1
        # enter13
        elif (
            key == 100 and selection < 1
        ):  # Dキー (cv2.KEY_DOWN_ARROW のキーコードに置き換えてください)
            selection += 1
        elif key == 13:
            # print("Enter key is pressed")
            break
    return selection


# if __name__ == "__main__":
# P2 = get_projection_matrix(0)
# P1 = get_projection_matrix(1)


# ローカル座標系に変換する関数
def compute_local_torque(torque_global, link_vec, parent_vec=None):
    """
    グローバル座標系のトルクをリンク基準の右手系に変換する。

    - z 軸: リンク方向。
    - y 軸: 親リンク parent_vec が与えられた場合は parent×z を採用し、
      前腕と上腕の法線（肘面）など、両リンクに直交する軸を優先する。
      parent_vec が無い/退化する場合は従来の基準軸外積にフォールバック。
    - x 軸: y×z。
    """
    if not np.all(np.isfinite(link_vec)):
        return torque_global
    norm_link = np.linalg.norm(link_vec)
    if norm_link < 1e-12:
        return torque_global

    z_axis = link_vec / norm_link

    # 優先: 親リンクとの平面法線を y とする（例: 肘で前腕・上腕に直交）
    if parent_vec is not None and np.all(np.isfinite(parent_vec)):
        parent_norm = np.linalg.norm(parent_vec)
        if parent_norm >= 1e-12:
            parent_unit = parent_vec / parent_norm
            y_candidate = np.cross(parent_unit, z_axis)
            y_norm = np.linalg.norm(y_candidate)
            if y_norm >= 1e-6:
                y_axis = y_candidate / y_norm
                x_candidate = np.cross(y_axis, z_axis)
                x_norm = np.linalg.norm(x_candidate)
                if x_norm >= 1e-6:
                    x_axis = x_candidate / x_norm
                    rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
                    return rotation.T @ torque_global

    # グローバル軸との外積でx軸を構成し、特異姿勢を避ける
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
        # どうしても決まらない場合は元の値を返す
        return torque_global

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-12:
        return torque_global
    y_axis /= y_norm

    rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
    torque_local = rotation.T @ torque_global
    return torque_local


class PushCycleDetector:
    def __init__(
        self, initial_z, threshold=0.015, velocity_epsilon=0.01, min_interval=10
    ):
        """
        Parameters
        ----------
        initial_z : float
            初期座標（安定座位姿勢の左肩z）
        threshold : float
            初期zとの差がこの値より小さくなったら候補（例：0.07）
        velocity_epsilon : float
            z軸速度がこの値以内なら「速度ゼロに近い」と判定
        min_interval : int
            サイクル間の最小フレーム数（誤検出防止）
        """
        self.initial_z = initial_z
        self.threshold = threshold
        self.velocity_epsilon = velocity_epsilon
        self.min_interval = min_interval

        self.prev_z = None
        self.last_cycle_frame = -min_interval
        self.cycles = []

    def update(self, z_current, frame_idx):
        if self.prev_z is None:
            self.prev_z = z_current
            return False

        # z軸速度（前フレームとの差分）
        dz = z_current - self.prev_z
        self.prev_z = z_current

        # 閾値条件 & 速度条件
        z_condition = z_current < self.initial_z + self.threshold
        velocity_condition = abs(dz) < self.velocity_epsilon

        if z_condition and velocity_condition:
            if frame_idx - self.last_cycle_frame > self.min_interval:
                self.last_cycle_frame = frame_idx
                self.cycles.append(frame_idx)
                print(
                    f"[Cycle Detected] Frame: {frame_idx}, z: {z_current:.3f}, z_int: {self.initial_z:.4f}"
                )
                return True

        return False
