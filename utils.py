import textwrap

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


# direct linear transform
def DLT(P1, P2, point1, point2):

    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    from scipy import linalg

    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3, 0:3] / Vh[3, 3]


def read_camera_parameters(camera_id, savefolder=folder_path + "\\camera_parameters\\"):

    inf = open(savefolder + "c" + str(camera_id) + ".dat", "r")
    # print("camera parameters read")
    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)


def read_rotation_translation(
    camera_id, savefolder=folder_path + "\\camera_parameters\\"
):

    inf = open(savefolder + "rot_trans_c" + str(camera_id) + ".dat", "r")

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)


def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis=1)
    else:
        return np.concatenate([pts, [1]], axis=0)


def get_projection_matrix(camera_id, file_mode):
    # read camera parameters
    if file_mode:
        cmtx, dist = read_camera_parameters(
            camera_id, folder_path + "\\camera_parameters\\Param_for_MYvideo\\"
        )
        rvec, tvec = read_rotation_translation(
            camera_id, folder_path + "\\camera_parameters\\Param_for_MYvideo\\"
        )
    else:
        cmtx, dist = read_camera_parameters(camera_id)
        rvec, tvec = read_rotation_translation(camera_id)

    # calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3, :]
    return P


def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, "w")

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + " " + str(kpt[1]) + " ")
            else:
                fout.write(str(kpt[0]) + " " + str(kpt[1]) + " " + str(kpt[2]) + " ")

        fout.write("\n")
    fout.close()


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

    frame0_keypoints = []
    frame1_keypoints = []
    if results0.pose_landmarks:
        for i, landmark in enumerate(results0.pose_landmarks.landmark):
            if i not in pose_keypoints:
                continue  # only save keypoints that are indicated in pose_keypoints
            pxl_x = landmark.x * frame0.shape[1]
            pxl_y = landmark.y * frame0.shape[0]
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            cv.circle(
                frame0, (pxl_x, pxl_y), 3, (0, 0, 255), -1
            )  # add keypoint detection points into figure
            label = str(i)  # ループのインデックスを文字列に変換
            cv.putText(
                frame0,
                label,
                (pxl_x + 5, pxl_y - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            kpts = [pxl_x, pxl_y]
            frame0_keypoints.append(kpts)
            # print(i,pxl_x,pxl_y)
        # print("keypoints detected")
    else:
        # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
        frame0_keypoints = [[-1, -1]] * len(pose_keypoints)
    if results1.pose_landmarks:
        for i, landmark in enumerate(results1.pose_landmarks.landmark):
            if i not in pose_keypoints:
                continue
            pxl_x = landmark.x * frame1.shape[1]
            pxl_y = landmark.y * frame1.shape[0]
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            cv.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
            kpts = [pxl_x, pxl_y]
            frame1_keypoints.append(kpts)

    else:
        # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
        frame1_keypoints = [[-1, -1]] * len(pose_keypoints)
        print("no keypoints detected111")

    return frame0_keypoints, frame1_keypoints


def calculate_3d_keypoints(frame0_keypoints, frame1_keypoints, P0, P1, pose_keypoints):
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
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(folder_path + "\\meiryo\\meiryo.ttc", font_size)
    wrapped_text = textwrap.fill(text, width=line_width)
    draw.text(position, wrapped_text, font=font, fill=color)
    return np.array(img_pil)


def display_choices(question, a, b):
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
    cv.destroyAllWindows()


# if __name__ == "__main__":
# P2 = get_projection_matrix(0)
# P1 = get_projection_matrix(1)


# ローカル座標系に変換する関数
def compute_local_torque(torque_global, link_vec):
    """
    グローバル座標系で与えられたトルクベクトルを、リンクの方向に基づいて
    ローカル座標系に変換する関数。

    リンクベクトルの向きに基づいて z 軸および y 軸方向の回転行列を構築し、
    それらを合成して回転行列 R を得る。これによりトルクをローカル座標系へ変換する。

    Parameters
    ----------
    torque_global : ndarray
        グローバル座標系におけるトルクベクトル（shape: 3,）。
    link_vec : ndarray
        リンク方向ベクトル（shape: 3,）。基準となる軸の向きを定義する。

    Returns
    -------
    torque_local : ndarray
        ローカル座標系に変換されたトルクベクトル（shape: 3,）。

    Notes
    -----
    - 回転順序は z軸 → y軸。
    - z軸回転 `phi` はリンクの xy 平面上の方向。
    - y軸回転 `theta` はリンクの z 軸に対する傾き。
    - 回転は右手系を想定。
    """
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
