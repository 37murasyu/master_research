import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=no-member
import cv2 as cv
import pandas as pd
from config import w, folder_path

# Small numeric epsilon for numerical stability
EPS = 1e-12


def calculate_inertia_tensor(k, mass, l):
    """
    質量と長さに基づいて、対象部位の慣性テンソルを計算する関数。

    CSVファイルから部位ごとの推定係数（a, b, c）を読み込み、以下の式を用いて
    各軸方向（x, y, z）の慣性モーメントを計算し、対角慣性テンソルを構築する：

        I = a * w + b * l + c

    Parameters
    ----------
    k : int
        係数データを取得するCSVの行インデックス（部位を表す）。
    mass : float
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

    row = pd.read_csv(csv_path).iloc[k]
    coeffs = np.array([row.iloc[i] for i in range(1, 10)], dtype=np.float64)
    coeffs *= 1e-4  # CSV values are in units of 1e-4 kg*m^2
    ax, bx, cx, ay, by, cy, az, bz, cz = coeffs
    I_tensor = np.diag([
        ax * mass + bx * l + cx,
        ay * mass + by * l + cy,
        az * mass + bz * l + cz,
    ])
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

    if not part_data:
        return np.zeros(3), np.zeros(3), "unknown"

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

    # NaN/None対策
    I = np.zeros((3, 3)) if I is None else I
    omega = np.zeros(3) if omega is None else omega
    dot_omega = np.zeros(3) if dot_omega is None else dot_omega
    dot_dot_pg = np.zeros(3) if dot_dot_pg is None else dot_dot_pg

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
        data_list = storage.get_data(part_j)
        if not data_list:
            torques.append((np.zeros(3), part_j))
            continue
        data_j = data_list[-1]
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
            y.pop(0)
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
    単純なオイラー積分: current_value + a*dt を返す。
    """
    return current_value + a * dt

# インパルス計算関数（compute_impulse と同一）
def compute_impulse(series: pd.Series, dt: float):
    arr = series.to_numpy()
    pos_imp = arr[arr > 0].sum() * dt
    neg_imp = arr[arr < 0].sum() * dt
    return pos_imp, neg_imp


# ====== Native dynamics acceleration (optional, direct ctypes loader to avoid hard import) ======
_NDLL = None
_NDYN_MF = None
_NDYN_TAU = None
_NDYN_LPF = None
_NDYN_TRIANG = None

def _load_native_dynamics() -> bool:  # pragma: no cover - thin loader
    global _NDLL, _NDYN_MF, _NDYN_TAU, _NDYN_LPF, _NDYN_TRIANG
    if _NDYN_MF is not None and _NDYN_TAU is not None and _NDYN_LPF is not None and _NDYN_TRIANG is not None:
        return True
    import os as _os
    import ctypes as _ct
    from ctypes import c_int as _c_int, c_double as _c_double, POINTER as _POINTER
    here = _os.path.dirname(__file__)
    _trace = _os.getenv('NDYN_TRACE', '0') not in ('0','false','False')
    candidates = [
        _os.path.join(here, 'native_dynamics', 'build', 'Release', 'native_dynamics.dll'),
        _os.path.join(here, 'native_dynamics', 'build', 'Debug', 'native_dynamics.dll'),
        _os.path.join(here, 'native_dynamics.dll'),
    ]
    for p in candidates:
        if _os.path.isfile(p):
            try:
                _NDLL = _ct.CDLL(p)
                break
            except OSError:
                _NDLL = None
    if _NDLL is None:
        if _trace:
            print('[NDYN] DLL not found; falling back to numpy')
        return False
    # set prototypes
    _NDLL.dyn_compute_mf_batch.argtypes = [
        _c_int,
        _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double),
        _POINTER(_c_double), _POINTER(_c_double),
    ]
    _NDLL.dyn_compute_mf_batch.restype = _c_int

    _NDLL.dyn_compute_tau_chain.argtypes = [
        _c_int,
        _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double),
        _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double), _POINTER(_c_double),
    ]
    _NDLL.dyn_compute_tau_chain.restype = _c_int

    _NDLL.dyn_lpf_exp_fb.argtypes = [
        _c_int,
        _POINTER(_c_double),
        _c_double,
        _c_double,
        _c_int,
        _POINTER(_c_double),
    ]
    _NDLL.dyn_lpf_exp_fb.restype = _c_int

    _NDLL.dyn_triangulate_transform_batch.argtypes = [
        _c_int,
        _POINTER(_c_double),
        _POINTER(_c_double),
        _POINTER(_c_double),
        _POINTER(_c_double),
        _c_double,
        _POINTER(_c_double),
    ]
    _NDLL.dyn_triangulate_transform_batch.restype = _c_int

    _NDYN_MF = _NDLL.dyn_compute_mf_batch
    _NDYN_TAU = _NDLL.dyn_compute_tau_chain
    _NDYN_LPF = _NDLL.dyn_lpf_exp_fb
    _NDYN_TRIANG = _NDLL.dyn_triangulate_transform_batch
    if _trace:
        try:
            _path = getattr(_NDLL, '_name', 'native_dynamics.dll')
        except Exception:
            _path = 'native_dynamics.dll'
        print(f'[NDYN] loaded: {_path}')
    return True


def compute_MF_batch_native(I_batch: np.ndarray,
                            m_batch: np.ndarray,
                            omega: np.ndarray,
                            dot_omega: np.ndarray,
                            ddpg: np.ndarray,
                            g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute M and F for a batch using native DLL if available; fall back to numpy.

    Shapes:
      I_batch: (N,3,3)
      m_batch: (N,)
      omega, dot_omega, ddpg: (N,3)
      g: (3,)
    Returns: (M: (N,3), F: (N,3))
    """
    import os as _os
    _trace = _os.getenv('NDYN_TRACE', '0') not in ('0','false','False')
    if _load_native_dynamics():
        try:
            from ctypes import c_double as _c_double, POINTER as _POINTER
            I_c = np.ascontiguousarray(I_batch, dtype=np.float64)
            m_c = np.ascontiguousarray(m_batch, dtype=np.float64)
            w_c = np.ascontiguousarray(omega, dtype=np.float64)
            dw_c = np.ascontiguousarray(dot_omega, dtype=np.float64)
            a_c = np.ascontiguousarray(ddpg, dtype=np.float64)
            g_c = np.ascontiguousarray(g, dtype=np.float64)
            N = int(I_c.shape[0])
            M_out = np.empty((N, 3), dtype=np.float64)
            F_out = np.empty((N, 3), dtype=np.float64)
            ret = _NDYN_MF(
                N,
                I_c.ctypes.data_as(_POINTER(_c_double)),
                m_c.ctypes.data_as(_POINTER(_c_double)),
                w_c.ctypes.data_as(_POINTER(_c_double)),
                dw_c.ctypes.data_as(_POINTER(_c_double)),
                a_c.ctypes.data_as(_POINTER(_c_double)),
                g_c.ctypes.data_as(_POINTER(_c_double)),
                M_out.ctypes.data_as(_POINTER(_c_double)),
                F_out.ctypes.data_as(_POINTER(_c_double)),
            )
            if ret == 0:
                if _trace:
                    print(f'[NDYN:MF] native ok (N={N})')
                return M_out, F_out
        except (OSError, RuntimeError, ValueError, AttributeError):
            if _trace:
                print('[NDYN:MF] native failed; fallback to numpy')
    # numpy fallback
    N = I_batch.shape[0]
    M = np.empty((N, 3), dtype=np.float64)
    F = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        Ii = I_batch[i]
        wi = omega[i]
        dwi = dot_omega[i]
        ai = ddpg[i]
        Iw = Ii @ wi
        M[i] = Ii @ dwi + np.cross(wi, Iw)
        F[i] = m_batch[i] * (ai - g)
    return M, F


def compute_tau_chain_native(Ms: np.ndarray,
                             Fs: np.ndarray,
                             r_gs: np.ndarray,
                             p1s: np.ndarray,
                             tau_E: np.ndarray,
                             f_E: np.ndarray,
                             r_x: np.ndarray) -> np.ndarray:
    """Compute joint torques for a chain using native DLL if available; fallback to numpy.

    tau_j = sum_{i>=j} M_i + sum_{i>=j} ((r_gs[i]-p1s[j]) x F_i) - tau_E - ((r_x - p1s[j]) x f_E)
    """
    import os as _os
    _trace = _os.getenv('NDYN_TRACE', '0') not in ('0','false','False')
    if _load_native_dynamics():
        try:
            from ctypes import c_double as _c_double, POINTER as _POINTER
            Ms_c = np.ascontiguousarray(Ms, dtype=np.float64)
            Fs_c = np.ascontiguousarray(Fs, dtype=np.float64)
            r_gs_c = np.ascontiguousarray(r_gs, dtype=np.float64)
            p1s_c = np.ascontiguousarray(p1s, dtype=np.float64)
            tau_E_c = np.ascontiguousarray(tau_E, dtype=np.float64)
            f_E_c = np.ascontiguousarray(f_E, dtype=np.float64)
            r_x_c = np.ascontiguousarray(r_x, dtype=np.float64)
            N = int(Ms_c.shape[0])
            tau_out = np.empty((N, 3), dtype=np.float64)
            ret = _NDYN_TAU(
                N,
                Ms_c.ctypes.data_as(_POINTER(_c_double)),
                Fs_c.ctypes.data_as(_POINTER(_c_double)),
                r_gs_c.ctypes.data_as(_POINTER(_c_double)),
                p1s_c.ctypes.data_as(_POINTER(_c_double)),
                tau_E_c.ctypes.data_as(_POINTER(_c_double)),
                f_E_c.ctypes.data_as(_POINTER(_c_double)),
                r_x_c.ctypes.data_as(_POINTER(_c_double)),
                tau_out.ctypes.data_as(_POINTER(_c_double)),
            )
            if ret == 0:
                if _trace:
                    print(f'[NDYN:TAU] native ok (N={N})')
                return tau_out
        except (OSError, RuntimeError, ValueError, AttributeError):
            if _trace:
                print('[NDYN:TAU] native failed; fallback to numpy')
    # numpy fallback
    N = Ms.shape[0]
    tau = np.zeros((N, 3), dtype=np.float64)
    # cumulative M from the end
    cumM = np.zeros((N, 3), dtype=np.float64)
    cumM[-1] = Ms[-1]
    for j in range(N-2, -1, -1):
        cumM[j] = cumM[j+1] + Ms[j]
    for j in range(N):
        t = cumM[j].copy()
        p1j = p1s[j]
        for i in range(j, N):
            rdiff = r_gs[i] - p1j
            t += np.cross(rdiff, Fs[i])
        t -= tau_E
        t -= np.cross(r_x - p1j, f_E)
        tau[j] = t
    return tau


def compute_lpf_exp_fb_native(x: np.ndarray,
                              dt: float,
                              fc: float,
                              passes: int = 2) -> np.ndarray:
    """Exponential forward-backward LPF using native DLL if available.

    Args:
      x: input signal (1-D)
      dt: sampling period [s]
      fc: cutoff frequency [Hz]
      passes: forward-backward repetitions
    """
    import os as _os
    _trace = _os.getenv('NDYN_TRACE', '0') not in ('0','false','False')
    x_c = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1), dtype=np.float64)
    n = int(x_c.shape[0])
    if n < 2:
        return x_c.copy()

    if _load_native_dynamics():
        try:
            from ctypes import c_double as _c_double, c_int as _c_int, POINTER as _POINTER
            y_out = np.empty_like(x_c)
            ret = _NDYN_LPF(
                n,
                x_c.ctypes.data_as(_POINTER(_c_double)),
                _c_double(float(dt)),
                _c_double(float(fc)),
                _c_int(int(max(1, passes))),
                y_out.ctypes.data_as(_POINTER(_c_double)),
            )
            if ret == 0:
                if _trace:
                    print(f'[NDYN:LPF] native ok (N={n}, dt={dt:.6f}, fc={fc:.3f}, passes={passes})')
                return y_out
        except (OSError, RuntimeError, ValueError, AttributeError):
            if _trace:
                print('[NDYN:LPF] native failed; fallback to numpy')

    # numpy fallback (same algorithm)
    rc = 1.0 / (2.0 * np.pi * max(1e-9, float(fc)))
    alpha = float(dt) / (rc + float(dt))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    y = x_c.copy()
    for _ in range(max(1, int(passes))):
        fwd = np.empty_like(y)
        fwd[0] = y[0]
        for i in range(1, n):
            fwd[i] = fwd[i - 1] + alpha * (y[i] - fwd[i - 1])
        bwd = np.empty_like(y)
        bwd[-1] = fwd[-1]
        for i in range(n - 2, -1, -1):
            bwd[i] = bwd[i + 1] + alpha * (fwd[i] - bwd[i + 1])
        y = bwd
    return y


def compute_triangulate_transform_native(P0: np.ndarray,
                                         P1: np.ndarray,
                                         keypoints0: np.ndarray,
                                         keypoints1: np.ndarray,
                                         scale: float = 0.01) -> np.ndarray:
    """Batch triangulate + transform via native DLL if available.

    Returns transformed points with shape (N, 3). Invalid points become NaN.
    Fallback uses OpenCV triangulatePoints batch + same transform.
    """
    import os as _os
    _trace = _os.getenv('NDYN_TRACE', '0') not in ('0','false','False')

    k0 = np.asarray(keypoints0, dtype=np.float64)
    k1 = np.asarray(keypoints1, dtype=np.float64)
    if k0.ndim != 2 or k1.ndim != 2 or k0.shape[1] != 2 or k1.shape[1] != 2:
        raise ValueError('keypoints must be shaped (N, 2)')
    if k0.shape[0] != k1.shape[0]:
        raise ValueError('keypoints length mismatch')

    n = int(k0.shape[0])
    out = np.full((n, 3), np.nan, dtype=np.float64)
    if n == 0:
        return out

    P0_c = np.ascontiguousarray(np.asarray(P0, dtype=np.float64).reshape(3, 4), dtype=np.float64)
    P1_c = np.ascontiguousarray(np.asarray(P1, dtype=np.float64).reshape(3, 4), dtype=np.float64)
    k0_c = np.ascontiguousarray(k0.reshape(n, 2), dtype=np.float64)
    k1_c = np.ascontiguousarray(k1.reshape(n, 2), dtype=np.float64)

    if _load_native_dynamics():
        try:
            from ctypes import c_double as _c_double, c_int as _c_int, POINTER as _POINTER
            ret = _NDYN_TRIANG(
                _c_int(n),
                P0_c.ctypes.data_as(_POINTER(_c_double)),
                P1_c.ctypes.data_as(_POINTER(_c_double)),
                k0_c.ctypes.data_as(_POINTER(_c_double)),
                k1_c.ctypes.data_as(_POINTER(_c_double)),
                _c_double(float(scale)),
                out.ctypes.data_as(_POINTER(_c_double)),
            )
            if ret == 0:
                if _trace:
                    valid = int(np.sum(np.all(np.isfinite(out), axis=1)))
                    print(f'[NDYN:TRI] native ok (N={n}, valid={valid})')
                return out
        except (OSError, RuntimeError, ValueError, AttributeError):
            if _trace:
                print('[NDYN:TRI] native failed; fallback to OpenCV batch')

    # OpenCV fallback
    valid_idx = [i for i in range(n) if np.all(np.isfinite(k0_c[i])) and np.all(np.isfinite(k1_c[i])) and k0_c[i, 0] >= 0 and k0_c[i, 1] >= 0 and k1_c[i, 0] >= 0 and k1_c[i, 1] >= 0]
    if not valid_idx:
        return out

    pts0 = np.array([[k0_c[i, 0], k0_c[i, 1]] for i in valid_idx], dtype=np.float64).T
    pts1 = np.array([[k1_c[i, 0], k1_c[i, 1]] for i in valid_idx], dtype=np.float64).T
    Xh = cv.triangulatePoints(P0_c, P1_c, pts0, pts1)
    wv = Xh[3, :]
    for j, idx in enumerate(valid_idx):
        wj = float(wv[j])
        if (not np.isfinite(wj)) or abs(wj) < 1e-12:
            continue
        Xj = (Xh[:3, j] / wj).astype(np.float64)
        if not np.all(np.isfinite(Xj)):
            continue
        out[idx, 0] = -Xj[0] * float(scale)
        out[idx, 1] = -Xj[2] * float(scale)
        out[idx, 2] = -Xj[1] * float(scale)
    return out

