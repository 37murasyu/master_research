import textwrap
import os
import time
import warnings
from functools import lru_cache

# pylint: disable=no-member
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
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


def read_camera_parameters(camera_id, savefolder=None):
    savefolder = savefolder or os.path.join(folder_path, "camera_parameters")
    path = os.path.join(savefolder, f"c{camera_id}.dat")
    with open(path, "r", encoding="utf-8") as inf:
        _ = inf.readline()
        cmtx = [[float(en) for en in inf.readline().split()] for _ in range(3)]
        _ = inf.readline()
        dist = [float(en) for en in inf.readline().split()]
    return np.array(cmtx), np.array([dist])


def read_rotation_translation(camera_id, savefolder=None):
    savefolder = savefolder or os.path.join(folder_path, "camera_parameters")
    path = os.path.join(savefolder, f"rot_trans_c{camera_id}.dat")
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


def get_projection_matrix(camera_id, file_mode, base_dir=None):
    if base_dir:
        base = base_dir
    else:
        base = os.path.join(folder_path, "camera_parameters")
        if file_mode:
            base = os.path.join(base, "Param_for_MYvideo")
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
            抽出対象となる関節インデックスの集合。これに含まれるキーポイントだけを抽出・描画する。
            **戻り値の並びはこのリストの順ではなくランドマーク ID の昇順**。
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
        # ランドマーク ID の昇順で返す。pose_keypoints の並び順ではない。
        # 元実装 TemugeB/bodypose3d は enumerate で昇順に走査しており、リストの
        # 並びは「どの点を使うか」のフィルタでしかない。ここをリスト順で回すと
        # 3D 点列の並びが part_calculations の前提とずれ、8 リンク中 7 本が
        # 体を斜めに横切るベクトルになる（再検算 R-1）。
        if not results.pose_landmarks:
            return [[-1, -1]] * len(pose_keypoints)
        out = []
        for pid in sorted(pose_keypoints):
            lm = results.pose_landmarks.landmark[pid]
            px = int(round(lm.x * frame.shape[1]))
            py = int(round(lm.y * frame.shape[0]))
            if draw_kpts:
                cv.circle(frame, (px, py), 3, (0, 0, 255), -1)
                cv.putText(frame, str(pid), (px + 5, py - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            out.append([px, py])
        return out

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


@lru_cache(maxsize=512)
def _glyph_sprite(ch: str, font_size: int, color: tuple):
    """1 文字を RGBA スプライトにしてキャッシュする。

    文字列全体でキャッシュすると数値が変わるたびに描き直しになるので、文字単位で持つ。
    ラベル＋数字なら数十エントリで飽和し、以降はすべてキャッシュに当たる。

    戻り値: (premul (h,w,3) f32, inv_a (h,w,3) f32, ox, oy, advance)
      premul = color × alpha、inv_a = 1 − alpha を前計算しておき、合成を 2 演算にする。
    """
    from app.core.resources import japanese_font

    font = japanese_font(int(font_size))
    try:
        adv = float(font.getlength(ch))
    except Exception:
        adv = float(font_size)
    try:
        x0, y0, x1, y1 = font.getbbox(ch)
    except Exception:
        x0, y0, x1, y1 = 0, 0, int(adv), int(font_size * 1.2)
    w = max(1, int(x1 - x0) + 1)
    h = max(1, int(y1 - y0) + 1)
    spr = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(spr).text((-x0, -y0), ch, font=font, fill=(*(int(c) for c in color[:3]), 255))
    rgba = np.asarray(spr, dtype=np.uint8)
    a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    # inv_a は (h,w,1) のままだと合成時に末尾軸が stride 0 になり、numpy がベクトル化ループを
    # 使えず 3 要素ずつのバッファリング処理に落ちる。(h,w,3) に展開しておくと合成が 4〜8 倍速い。
    inv_a = np.ascontiguousarray(np.broadcast_to(1.0 - a, (a.shape[0], a.shape[1], 3)))
    return (rgba[:, :, :3].astype(np.float32) * a, inv_a, int(x0), int(y0), adv)


def prewarm_text_jp(chars: str, font_size: int, color) -> None:
    """使う文字のスプライトを先に作っておく。初回フレームでフォント読み込み（約 15 ms）と
    グリフ描画がまとめて走ると 1 フレーム落ちるため、起動時に呼ぶ。"""
    for ch in chars:
        _glyph_sprite(ch, int(font_size), tuple(color))


def draw_text_jp(img, text, position, font_size, color, line_width=20):
    """HxWx3 の uint8 画像に日本語テキストを**その場で**描き、同じ配列を返す。

    フレーム全体を PIL へ往復させず、文字ごとのスプライト（キャッシュ）を合成する。
    折り返しと行送りは PIL（``put_text_jp`` の従来の実装）と同じ: 常に ``textwrap.fill`` を通し、
    行送りは「A の高さ + 4」。色は配列のチャネル順のまま書く（OpenCV の画像なら BGR）。

    かつて ``master_research_code.py`` の ``_blit_label`` が同じことを別に実装しており、
    折り返し（長いときだけ）と行送り（文字サイズ × 1.25）が食い違っていた（KNOWN_ISSUES §4-2）。
    """
    from app.core.resources import japanese_font

    font_size = int(font_size)
    color = tuple(int(c) for c in color[:3])
    height, width = img.shape[:2]
    line_spacing = japanese_font(font_size).getbbox("A")[3] + 4
    for row, line in enumerate(textwrap.fill(text, width=line_width).split("\n")):
        pen_x = 0.0
        pen_y = int(position[1]) + row * line_spacing
        for ch in line:
            premul, inv_a, ox, oy, adv = _glyph_sprite(ch, font_size, color)
            if ch != " ":
                sh, sw = premul.shape[:2]
                x = int(position[0]) + int(pen_x) + ox
                y = pen_y + oy
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(width, x + sw), min(height, y + sh)
                if x1 > x0 and y1 > y0:
                    sx, sy = x0 - x, y0 - y
                    dst = img[y0:y1, x0:x1]
                    blended = dst * inv_a[sy:sy + y1 - y0, sx:sx + x1 - x0] + premul[sy:sy + y1 - y0, sx:sx + x1 - x0]
                    # PIL の合成は四捨五入（MULDIV255）。切り捨てると灰色の背景で 1 ずれる
                    np.copyto(dst, (blended + 0.5).astype(np.uint8))
            pen_x += adv
    return img


def put_text_jp(img, text, position, font_size, color, line_width):
    """
    OpenCV画像に日本語テキストを描画する関数。

    OpenCV形式の画像に対して、日本語フォントに対応したテキストを指定位置に描画し、
    改行幅を考慮して整形した後、新たな画像をOpenCV形式で返す（入力は変えない）。

    3 チャネルの uint8 画像は ``draw_text_jp``（文字ごとのスプライト合成）で描く。
    それ以外（グレースケールなど）は従来どおり PIL で描く。

    Parameters:
        img (numpy.ndarray): OpenCV形式の入力画像。
        text (str): 描画する日本語テキスト。
        position (tuple): テキストの描画位置（x, y）。
        font_size (int): フォントサイズ。
        color (tuple): テキストの色（配列のチャネル順。OpenCV の画像なら B, G, R）。
        line_width (int): 1行あたりの最大文字数（改行幅）。

    Returns:
        numpy.ndarray: テキストが描画されたOpenCV形式の画像。
    """
    if img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
        return draw_text_jp(img.copy(), text, position, font_size, color, line_width)

    # フォントの取得・キャッシュ・欠落時の扱いは resources に集約してある。
    from app.core.resources import japanese_font

    img_pil = Image.fromarray(img)
    ImageDraw.Draw(img_pil).text(
        position, textwrap.fill(text, width=line_width), font=japanese_font(int(font_size)), fill=color)
    return np.array(img_pil)


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


class LocalFrameFallbackWarning(RuntimeWarning):
    """局所座標系を作れず、全体座標の値をそのまま返したことを知らせる。

    リンクが非有限・長さ 0 のときに出る。値はそのまま返す（2026-09-23 に決定、
    KNOWN_ISSUES §5-4）。100 N·m 級の全体座標の値が ``*_local_*`` 列に黙って紛れ込むのを
    見逃さないための警告で、Python の既定では呼び出し箇所ごとに 1 回だけ表示される。
    """


def _global_value_with_warning(torque_global, reason):
    warnings.warn(
        f"局所座標系を作れない（{reason}）ので、全体座標の値をそのまま返す",
        LocalFrameFallbackWarning,
        stacklevel=3,
    )
    return torque_global


# ローカル座標系に変換する関数
def compute_joint_power(torque_global, omega_link, omega_parent, link_vec, parent_vec=None, up_axis=None):
    """関節の仕事率を、局所トルクの y 軸まわりで求める。

        P = τ_y × ((ω_link − ω_parent) · y)

    y 軸は ``compute_local_torque`` と同じ作り方にする（z = link_vec、y = parent_vec × z、
    parent_vec が無い／退化するときは全体座標の基準軸から）。τ と ω を**同じ軸**に射影するので、
    左右を鏡映すると τ_y と ω_y は必ず揃って符号を変え、積 P は変わらない。

    かつてスコア経路（``compute_cycle_energy_elbow_wrist.py``）は、τ_y に「+Y まわりの角度」や
    「水平面からの傾き」の微分を掛けていた。軸の作り方が τ と別なので、鏡映で片方だけ
    符号が反転し、左右で逆の相（押し出しと戻し）を積算していた。

    Parameters
    ----------
    torque_global : ndarray, shape (3,)
        関節トルク（全体座標）。
    omega_link : ndarray, shape (3,)
        link_vec の部位の角速度（全体座標）。
    omega_parent : ndarray, shape (3,) or None
        親部位の角速度（全体座標）。None は親が動かないことを表す
        （例: アームレストを押す手を固定端とみなしたときの手首）。
    link_vec, parent_vec, up_axis : ndarray, shape (3,)
        ``compute_local_torque`` と同じ。

    Returns
    -------
    float
        仕事率 [W]。
    """
    omega_rel = np.asarray(omega_link, dtype=np.float64)
    if omega_parent is not None:
        omega_rel = omega_rel - np.asarray(omega_parent, dtype=np.float64)
    # compute_local_torque の中身は「局所座標系への回転」なので角速度にもそのまま使える。
    # τ と ω を必ず同じ関数・同じ引数で射影する。軸を作れず全体座標のまま返るとき
    # （KNOWN_ISSUES §5-4）も、両者の扱いが揃う。
    tau_local = compute_local_torque(np.asarray(torque_global, dtype=np.float64), link_vec, parent_vec, up_axis)
    omega_local = compute_local_torque(omega_rel, link_vec, parent_vec, up_axis)
    return float(tau_local[1] * omega_local[1])


def compute_local_torque(torque_global, link_vec, parent_vec=None, up_axis=None):
    """
    グローバル座標系のトルクをリンク基準の右手系に変換する。

    - z 軸: リンク方向。
    - y 軸: 親リンク parent_vec が与えられた場合は parent×z を採用し、
      前腕と上腕の法線（肘面）など、両リンクに直交する軸を優先する。
      parent_vec が無い/退化する場合は基準軸との外積にフォールバックする。
      このとき y は「基準軸のうちリンクに直交する成分」になる。
    - x 軸: y×z。

    up_axis は、フォールバックで最初に試す基準軸（鉛直上向き）。省略すると全体座標の z。
    z が上のリアルタイム経路では省略してよいが、y が鉛直で z が奥行きのカメラ座標
    （オフラインの入力 CSV）では重力の逆向きを渡すこと（KNOWN_ISSUES §1-5）。

    リンクが非有限・長さ 0 で軸を作れないときは、torque_global をそのまま返して
    LocalFrameFallbackWarning を出す（§5-4）。
    """
    if not np.all(np.isfinite(link_vec)):
        return _global_value_with_warning(torque_global, "リンクが非有限")
    norm_link = np.linalg.norm(link_vec)
    if norm_link < 1e-12:
        return _global_value_with_warning(torque_global, "リンクの長さが 0")

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
    if up_axis is not None:
        up = np.asarray(up_axis, dtype=np.float64)
        up_norm = np.linalg.norm(up)
        if np.all(np.isfinite(up)) and up_norm > 1e-12:
            reference_axes = (up / up_norm,) + reference_axes
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

    # 直交する 3 軸のうち 2 本と同時に |cos| >= 0.95 にはなれないので、ここへは来ないはず。
    if x_axis is None:
        return _global_value_with_warning(torque_global, "基準軸がすべてリンクと平行")

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-12:
        return _global_value_with_warning(torque_global, "y 軸が潰れた")
    y_axis /= y_norm

    rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
    torque_local = rotation.T @ torque_global
    return torque_local


class PushCycleDetector:
    def __init__(
        self,
        initial_z,
        threshold=0.015,
        velocity_epsilon=0.01,
        min_interval=10,
        mode='legacy',
        negative_down=True,
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
        self.mode = mode
        self.negative_down = bool(negative_down)

        self.prev_z = None
        self.last_cycle_frame = -min_interval
        self.cycles = []
        self._seen_drop = False

    def update(self, z_current, frame_idx):
        if self.prev_z is None:
            self.prev_z = z_current
            return False

        # z軸速度（前フレームとの差分）
        z_prev = self.prev_z
        dz = z_current - z_prev
        self.prev_z = z_current

        if self.mode == 'rise_to_rise':
            # たち下がり（drop）を一度経由した後に、立ち上がり境界（rise）でサイクル確定
            if self.negative_down:
                drop_cond = (z_current < self.initial_z - self.threshold) or (dz < -abs(self.velocity_epsilon))
                rise_level = self.initial_z - 0.25 * self.threshold
                rise_cross = (z_prev < rise_level <= z_current)
            else:
                drop_cond = (z_current > self.initial_z + self.threshold) or (dz > abs(self.velocity_epsilon))
                rise_level = self.initial_z + 0.25 * self.threshold
                rise_cross = (z_prev > rise_level >= z_current)

            if drop_cond:
                self._seen_drop = True

            if self._seen_drop and rise_cross:
                if frame_idx - self.last_cycle_frame > self.min_interval:
                    self.last_cycle_frame = frame_idx
                    self.cycles.append(frame_idx)
                    self._seen_drop = False
                    print(
                        f"[Cycle Detected] Frame: {frame_idx}, sig: {z_current:.3f}, sig_init: {self.initial_z:.4f} (rise_to_rise)"
                    )
                    return True
            return False

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
