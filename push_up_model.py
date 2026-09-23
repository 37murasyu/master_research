"""座位プッシュアップの逆動力学モデル。オフライン・USB・スマホの 3 経路で共有する。

モデル（KNOWN_ISSUES §2-1、2026-09-09 確定）:
    車椅子のアームレストを両手で押す動作を、手を外力との境界（固定端）とし、そこから
    近位 ── 前腕 → 上腕 ── を持ち上げる鎖として解く。肩より近位の体幹と頭は「肩に載る外力」
    に集約し、脚は床／フットレストが支えるので含めない。

かつては 3 経路がそれぞれ別に書いており、次のように食い違っていた。

- 重力: オフラインは奥行き方向（−z）に置いていた。入力 CSV は y が鉛直（§1-5）
- 関節の基準点: USB・スマホ経路は部位の並びがずれ、``wrist_R`` が右肘まわりのトルクだった（§5-7）
- 体幹荷重: USB・スマホ経路は下胴体に体重 60 kg を丸ごと渡しており、200〜300 N·m 出ていた（§5-8）
- 手首の局所軸: 肘の屈曲軸や全体座標の基準軸のままで、手首の軸になっていなかった（§5-1）
- 慣性テンソル: 部位固定系の対角テンソルを回さずに全体座標の ω に掛けていた（§2-3）

ここにモデルの約束をまとめ、各経路はこのモジュールを呼ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SEGMENT_MASS_FRACTIONS, SUPPORT_SHARE_DEFAULT, SUPPORTED_MASS_FRACTION, slot_of
from utils import compute_joint_power
from utils_dynamic import compute_MF_batch_native, compute_tau_chain_native

GRAVITY_MODES = ("axis", "trunk")

# 手首の軸を手のひらから作るのに要る、前腕と手のなす角の下限（sin）。約 14.5°。
# 手首がまっすぐに近いと、前腕 × 手の外積の向きが点のノイズで決まってしまう。
# プッシュアップ中の手首は 60〜90° 背屈しているので、普通はこの下限に掛からない。
MIN_WRIST_BEND_SIN = 0.25


# ---------------------------------------------------------------------------
# 重力（§1-5）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GravityEstimate:
    """体幹の向きから決めた重力。"""

    vector: np.ndarray    # 重力加速度 [m/s²]
    up: np.ndarray        # 採用した上向き（単位ベクトル、= −vector / |vector|）
    trunk_up: np.ndarray  # 体幹の上向き（初期フレームの中央値、単位ベクトル）
    lean_deg: float       # 体幹が採用した上向きから傾いている角度
    mode: str
    samples: int


def estimate_gravity(trunk_up, magnitude: float, mode: str = "axis") -> GravityEstimate:
    """初期フレームの体幹の上向き（腰中点 → 肩中点）から重力を決める。

    Parameters
    ----------
    trunk_up : array_like, shape (N, 3)
        フレームごとの体幹の上向き。非有限の行は捨てる。
    magnitude : float
        重力加速度の大きさ。
    mode : {"axis", "trunk"}
        "axis"（既定）は体幹の向きに最も近い座標軸を上とする。USB 経路の
        ``GRAVITY_AUTO_DETECT`` と同じ考え方で、カメラが水平に置かれていることを前提にする。
        座位の体幹は 10° 前後傾いており（入力 CSV の実測）、体幹そのものを鉛直とみなすと
        重力が同じだけ傾くため、こちらを既定にした。
        "trunk" は体幹の向きそのものを上とする（カメラが傾いている場合向け）。

    Raises
    ------
    ValueError
        使えるフレームが 1 つも無いとき、または mode が不正なとき。
    """
    if mode not in GRAVITY_MODES:
        raise ValueError(f"mode は {GRAVITY_MODES} のどれか: {mode!r}")
    ups = np.asarray(trunk_up, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(ups, axis=1)
    usable = np.all(np.isfinite(ups), axis=1) & (norms > 1e-9)
    if not usable.any():
        raise ValueError("体幹の向きを取れるフレームが無いので重力を決められない")
    # 単位ベクトルにしてから成分ごとの中央値を取る。1 フレームの外れ値に引きずられない。
    median = np.median(ups[usable] / norms[usable, None], axis=0)
    trunk = median / np.linalg.norm(median)

    if mode == "axis":
        axis = int(np.argmax(np.abs(trunk)))
        up = np.zeros(3)
        up[axis] = np.sign(trunk[axis])
    else:
        up = trunk.copy()
    lean = float(np.degrees(np.arccos(np.clip(np.dot(trunk, up), -1.0, 1.0))))
    return GravityEstimate(
        vector=-float(magnitude) * up, up=up, trunk_up=trunk,
        lean_deg=lean, mode=mode, samples=int(usable.sum()))


def trunk_up_vectors(l_shoulder, r_shoulder, l_hip, r_hip):
    """体幹の上向き = 肩中点 − 腰中点。(3,) でも (T, 3) でもよい。"""
    return 0.5 * (np.asarray(l_shoulder) + np.asarray(r_shoulder)) - 0.5 * (
        np.asarray(l_hip) + np.asarray(r_hip))


# ---------------------------------------------------------------------------
# 質量（§2-1、§2-2）
# ---------------------------------------------------------------------------


def torso_load_mass(body_mass: float, share: float = SUPPORT_SHARE_DEFAULT, torso_mass: float | None = None) -> float:
    """片腕が肩で支える体幹＋頭の質量 [kg]。

    既定は Winter の体幹 0.497 ＋頭頸 0.081（H-A、2026-09-23 に現行の比のまま確定）を両腕で等分。
    """
    total = body_mass * SUPPORTED_MASS_FRACTION if torso_mass is None else torso_mass
    return max(float(total), 0.0) * float(np.clip(share, 0.0, 1.0))


def hand_mass(body_mass: float) -> float:
    """手の質量 [kg]。"""
    return float(body_mass) * SEGMENT_MASS_FRACTIONS["hand"]


def hand_point(pinky, index):
    """手の代表点 = 小指と人差し指（の付け根）の中点。中手骨頭の代わりに使う。"""
    return 0.5 * (np.asarray(pinky, dtype=np.float64) + np.asarray(index, dtype=np.float64))


# ---------------------------------------------------------------------------
# 局所軸（§5-1）
# ---------------------------------------------------------------------------


def wrist_hand_mask(elbow, wrist, hand):
    """手首の軸を手のひらから作れるフレームなら True（手の点が有限で、手首が曲がっている）。"""
    forearm = np.asarray(elbow, dtype=np.float64) - np.asarray(wrist, dtype=np.float64)
    palm = np.asarray(wrist, dtype=np.float64) - np.asarray(hand, dtype=np.float64)
    cross = np.linalg.norm(np.cross(palm, forearm), axis=-1)
    scale = np.linalg.norm(palm, axis=-1) * np.linalg.norm(forearm, axis=-1)
    with np.errstate(invalid="ignore"):
        bent = cross >= MIN_WRIST_BEND_SIN * scale
    return np.all(np.isfinite(palm), axis=-1) & bent & (scale > 0)


def joint_axes(shoulder, elbow, wrist, hand=None, other_shoulder=None):
    """関節ごとの局所軸の材料 (link, parent) を返す。``utils.compute_local_torque`` にそのまま渡す。

    局所軸は z = link、y = parent × z。向きは鎖の向き（固定端の手から外へ）に揃える:

    - 手首: link = 前腕（手首 → 肘）、parent = 手（手の点 → 手首）。y は前腕と手に直交する
      手首の屈曲軸。手の点が無い・手首がまっすぐに近いフレームでは parent = 肘 − 肩 とし、
      y を肘と同じ屈曲軸（同じ向き）にする。手を固定端として前腕が回る面は腕の面なので、
      手のひらの向きが分からなくても屈曲の成分はこの軸で取れる
    - 肘: link = 上腕（肘 → 肩）、parent = 前腕（手首 → 肘）。y は腕の面の法線
    - 肩: link = 上腕（肩 → 肘）、parent = 肩の線（反対の肩 → 肩）。other_shoulder が無ければ
      parent = None（全体座標の基準軸にフォールバックする）

    すべて (3,) でも (T, 3) でもよい。hand は None か、行ごとに非有限を含んでよい。
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    elbow = np.asarray(elbow, dtype=np.float64)
    wrist = np.asarray(wrist, dtype=np.float64)

    forearm = elbow - wrist
    fallback = elbow - shoulder
    if hand is None:
        wrist_parent = fallback
    else:
        palm = wrist - np.asarray(hand, dtype=np.float64)
        usable = wrist_hand_mask(elbow, wrist, hand)
        wrist_parent = np.where(np.expand_dims(usable, -1), palm, fallback)

    shoulder_parent = None
    if other_shoulder is not None:
        shoulder_parent = shoulder - np.asarray(other_shoulder, dtype=np.float64)

    return {
        "wrist": (forearm, wrist_parent),
        "elbow": (shoulder - elbow, forearm),
        "shoulder": (elbow - shoulder, shoulder_parent),
    }


# USB・スマホ経路の点列（pose_keypoints の昇順）から片腕の局所軸を作る。
# 部位名は config.part_calculations のキー（左の上腕だけ歴史的に名前が違う）。
ARM_PARTS: dict[str, dict[str, str]] = {
    "R": {"upper_arm": "upper_arm_R", "forearm": "forearm_R"},
    "L": {"upper_arm": "up_arm_l", "forearm": "forearm_L"},
}
_OTHER_SIDE = {"R": "L", "L": "R"}


def arm_axes(points, side: str):
    """点列（pose_keypoints の昇順）から片腕の ``joint_axes`` を作る。

    関節はランドマーク名で指す。位置索引を直書きすると pose_keypoints に点を足したときに
    別の関節を指す（再検算 R-1 と同じ壊れ方）。手首の軸は手の点（小指・人差し指）の中点から作る。
    """
    def at(landmark: str) -> np.ndarray:
        return np.asarray(points[slot_of(landmark)], dtype=np.float64)

    return joint_axes(
        at(f"{side}_SHOULDER"), at(f"{side}_ELBOW"), at(f"{side}_WRIST"),
        hand=hand_point(at(f"{side}_PINKY"), at(f"{side}_INDEX")),
        other_shoulder=at(f"{_OTHER_SIDE[side]}_SHOULDER"),
    )


# ---------------------------------------------------------------------------
# 慣性テンソル（§2-3）
# ---------------------------------------------------------------------------


def inertia_about_link(inertia, link):
    """部位固定系の対角慣性テンソル（z がリンク軸）を、全体座標へ回す。

    2 点のリンクからは軸まわりの向き（ロール）が取れないので、横方向の 2 軸を平均した
    軸対称テンソルで近似する: I = I_t (E − u uᵀ) + I_l u uᵀ（u はリンクの単位ベクトル）。
    慣性係数表の上腕・前腕は横 2 軸の差が 4〜15% で、長軸は 2〜6 倍小さい。
    かつては回さずに全体座標の ω に掛けており、リンクの向き次第で長軸の値が横の回転に使われていた。

    リンクが非有限・長さ 0 のときは回さずに返す。
    """
    tensor = np.asarray(inertia, dtype=np.float64)
    link = np.asarray(link, dtype=np.float64)
    norm = np.linalg.norm(link)
    if not np.all(np.isfinite(link)) or norm < 1e-12:
        return tensor.copy()
    u = link / norm
    diag = np.diag(tensor)
    transverse = 0.5 * (diag[0] + diag[1])
    along = np.outer(u, u)
    return transverse * (np.eye(3) - along) + diag[2] * along


# ---------------------------------------------------------------------------
# 関節トルク（§2-1、§5-7、§5-8）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentState:
    """1 フレームの 1 部位の状態（全体座標）。"""

    inertia: np.ndarray   # 部位固定系の対角テンソル（z がリンク軸）
    mass: float
    omega: np.ndarray
    domega: np.ndarray
    com_acc: np.ndarray
    com: np.ndarray
    link: np.ndarray      # リンクの向き。慣性テンソルを回すのに使う（向きの正負は問わない）


def segment_from_storage(entry: dict, inertia, mass: float) -> SegmentState:
    """``BodyPartDataStorage`` の 1 件から部位の状態を作る。未計算の量は 0 とみなす。"""
    def vec(key):
        value = entry.get(key)
        return np.zeros(3) if value is None else np.asarray(value, dtype=np.float64)

    return SegmentState(
        inertia=np.asarray(inertia, dtype=np.float64), mass=float(mass),
        omega=vec("omega"), domega=vec("dot_omega"), com_acc=vec("dot_dot_pg"),
        com=vec("centroid"), link=vec("relative_position_vector"))


def push_up_torques(
    forearm: SegmentState,
    upper_arm: SegmentState,
    wrist,
    elbow,
    shoulder,
    gravity,
    load_mass: float,
    hand_mass_kg: float,
) -> dict[str, np.ndarray]:
    """1 フレーム・片腕の関節トルク（全体座標）を返す。キーは "wrist" / "elbow" / "shoulder"。

    - 手首・肘: 手を固定端に、前腕（関節 = 手首）→ 上腕（関節 = 肘）の鎖。体幹＋頭の荷重
      ``load_mass`` を肩に載せる
    - 肩: 体幹荷重は肩に載る外力なのでモデル上は肩まわりのモーメントを作らない。代わりに
      腕を肩から吊った鎖（上腕 → 前腕）の自重を出す。手は手首の質点とする（理論 1RM 仕事量の
      分母と同じ近似。§2-2 で分母に手を含めると決めたので、腕を振る側の鎖にも入れる）

    トルクは「関節がその先の部分に加えるトルク」: τ_j = Σ_{i≥j} [M_i + (r_i − p_j) × m_i(a_i − g)]
    − (r_x − p_j) × f_E。静止なら先の部分の重さを支えるモーメントになる。
    """
    g = np.asarray(gravity, dtype=np.float64)
    segments = (forearm, upper_arm)
    moments, forces = compute_MF_batch_native(
        np.stack([inertia_about_link(s.inertia, s.link) for s in segments]),
        np.array([s.mass for s in segments], dtype=np.float64),
        np.stack([s.omega for s in segments]),
        np.stack([s.domega for s in segments]),
        np.stack([s.com_acc for s in segments]),
        g,
    )
    coms = np.stack([s.com for s in segments])
    zero = np.zeros(3)
    wrist_base = compute_tau_chain_native(
        moments, forces, coms, np.stack([wrist, elbow]).astype(np.float64),
        zero, float(load_mass) * g, np.asarray(shoulder, dtype=np.float64))
    hanging = compute_tau_chain_native(
        moments[::-1].copy(), forces[::-1].copy(), coms[::-1].copy(),
        np.stack([shoulder, elbow]).astype(np.float64),
        zero, float(hand_mass_kg) * g, np.asarray(wrist, dtype=np.float64))
    return {"wrist": wrist_base[0], "elbow": wrist_base[1], "shoulder": hanging[0]}


def push_up_joint_powers(torques, axes, forearm: SegmentState, upper_arm: SegmentState, trunk_omega, up=None):
    """関節の仕事率 P = τ_y × (ω_外側 − ω_内側)·y を返す（キーは "wrist" / "elbow" / "shoulder"）。

    局所 y 軸は局所トルクと同じ（``joint_axes``）。鎖の外側と内側の部位の相対角速度を使う:

    - 手首: 前腕（手は固定端）
    - 肘: 上腕 − 前腕
    - 肩: 上腕 − 上胴体（両肩の線の角速度。無ければ 0 とみなす）

    かつて部位の絶対角速度との内積 τ·ω で、肘角を保ったまま腕を振るだけで仕事が出ていた
    （計画メモ A-4 (2)、H-B）。入力に非有限が混じる関節は 0 にする。
    """
    trunk = None if trunk_omega is None else np.asarray(trunk_omega, dtype=np.float64)
    relative = {
        "wrist": (forearm.omega, None),
        "elbow": (upper_arm.omega, forearm.omega),
        "shoulder": (upper_arm.omega, trunk),
    }
    powers = {}
    for joint, torque in torques.items():
        link, parent = axes[joint]
        omega, omega_parent = relative[joint]
        vectors = [torque, omega, link] + ([omega_parent] if omega_parent is not None else [])
        if any(not np.all(np.isfinite(v)) for v in vectors):
            powers[joint] = 0.0
            continue
        powers[joint] = compute_joint_power(torque, omega, omega_parent, link, parent, up)
    return powers
