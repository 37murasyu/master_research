"""合成の座位プッシュアップを、混成の計測フォルダ（``landmarks2d_*`` ほか）として書き出す。

再生（``app.hybrid.replay``）で計測の全体に通し、「押し上げ 1 回ごとに回が閉じる」「論文の閾値を跨ぐと状態が変わる」
ことを被験者なしで確かめるための入力。2026-09-23 の実機の記録は上下の動きが小さい区間が多く、その根拠にならない。

体は Mac のカメラ座標（cm、x 右・y 下・z 前）で作る。手（手首・指）は肘掛けに固定し、肩と腰が持ち上がる。
肘は上腕 30 cm・前腕 25 cm の 2 リンクの逆運動学で、肩と手首から決める（後ろ外側へ曲げる）。膝と足首は動かさない。
最初の ``still_s`` 秒は座って静止する（計測は先頭の窓で重力・基準の高さ・前腕長を決めるため）。

    python -m tools.synth_session --out ~/Documents/WheelchairTorque/hybrid/synth --reps 10 --lift-cm 13

既定の配置は、被写体まで 150 cm、Pixel を Mac の右 60 cm・同じ高さに置いて被写体へ向けたもの（肘での視線の角は約 22°）。
``--calibration <校正フォルダ>`` を与えると、その校正（実機の Mac と Pixel の位置関係と歪み）で投影する。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import cv2 as cv
import numpy as np

from app.hybrid.calibration_io import load_calibration, save_calibration
from app.hybrid.checkerboard import Board, Intrinsics, Stereo
from app.hybrid.recorder import Recorder
from app.net.protocol import LandmarkFrame
from config import pose_keypoints, slot_of

__all__ = ["UPPER_ARM_CM", "FOREARM_CM", "lift_cm_at", "pushup_body_cm", "write_session"]

UPPER_ARM_CM = 30.0
FOREARM_CM = 25.0
DEPTH_CM = 150.0
# 座った姿勢（持ち上げ 0）の点。カメラ座標 [cm]。被写体の左は画面の右（+x）
_SEATED = {
    "SHOULDER": (19.0, -5.0, 0.0),     # (x の絶対値, y, 奥行きの DEPTH_CM からの差)
    "WRIST": (26.0, 35.0, -8.0),       # 肘掛けの上
    "PINKY": (29.0, 43.0, -8.0),
    "INDEX": (24.0, 44.0, -10.0),
    "HIP": (12.0, 45.0, 5.0),
    "KNEE": (13.0, 32.0, -40.0),
    "ANKLE": (13.0, 38.0, -40.0),
}
_LIFTED = ("SHOULDER", "HIP")   # 持ち上げで上がる点（手・膝・足首は動かない）


def lift_cm_at(t: float, *, lift_cm: float = 13.0, period_s: float = 3.0, still_s: float = 2.0,
               up_fraction: float = 0.7) -> float:
    """時刻 t [s] の持ち上げの高さ [cm]。1 周期のうち up_fraction で上がって下り、残りは座って休む。"""
    if t < still_s:
        return 0.0
    phase = ((t - still_s) % period_s) / period_s
    if phase >= up_fraction:
        return 0.0
    return lift_cm * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase / up_fraction))


def _elbow(shoulder: np.ndarray, wrist: np.ndarray, outward: float) -> np.ndarray:
    """肩と手首から肘の位置（2 リンクの逆運動学）。肘は後ろ（+z）と外側へ曲げる。"""
    axis = wrist - shoulder
    d = float(np.linalg.norm(axis))
    u = axis / d
    reach = UPPER_ARM_CM + FOREARM_CM
    if d >= reach:   # 伸びきった腕（持ち上げが大きすぎる）。長さは保てないので直線に置く
        return shoulder + u * UPPER_ARM_CM
    p = (UPPER_ARM_CM ** 2 - FOREARM_CM ** 2 + d ** 2) / (2.0 * d)
    h = np.sqrt(max(UPPER_ARM_CM ** 2 - p ** 2, 0.0))
    bend = np.array([outward, 0.0, 1.0])
    bend = bend - np.dot(bend, u) * u
    bend /= np.linalg.norm(bend)
    return shoulder + p * u + h * bend


def pushup_body_cm(t: float, **profile) -> np.ndarray:
    """時刻 t の体の 16 点（``pose_keypoints`` の昇順、Mac のカメラ座標 cm）。profile は ``lift_cm_at`` の引数。"""
    lift = lift_cm_at(t, **profile)
    body = np.zeros((len(pose_keypoints), 3))
    for side, sign in (("L", 1.0), ("R", -1.0)):
        for part, (x, y, dz) in _SEATED.items():
            point = np.array([sign * x, y, DEPTH_CM + dz])
            if part in _LIFTED:
                point[1] -= lift   # y は下向き
            body[slot_of(f"{side}_{part}")] = point
        body[slot_of(f"{side}_ELBOW")] = _elbow(
            body[slot_of(f"{side}_SHOULDER")], body[slot_of(f"{side}_WRIST")], outward=sign * 0.6)
    return body


def _default_calibration(root: Path):
    """被写体まで 150 cm、Pixel を右 60 cm・同じ高さに置いて被写体へ向けた配置の校正を作る。"""
    k = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    intr = Intrinsics(k, np.array([0.05, -0.02, 0.0, 0.0, 0.0]), (1280, 720), 0.0)
    angle = np.arctan2(60.0, DEPTH_CM)
    R = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    T = (-R @ np.array([60.0, 0.0, 0.0])).reshape(3, 1)
    directory = save_calibration(intr, intr, Stereo(R, T, 0.0, [], list(range(12))), Board(), root=root,
                                 cameras=[{"kind": "mac", "device_id": "mac-synth"},
                                          {"kind": "pixel", "device_id": "pixel-synth"}])
    return load_calibration(directory)


def _extrinsics(calibration) -> tuple[np.ndarray, np.ndarray]:
    from utils import read_rotation_translation

    R, T = read_rotation_translation(1, str(calibration.directory))
    return np.asarray(R, dtype=float), np.asarray(T, dtype=float).reshape(3)


def _marks(points_cm: np.ndarray, intr, rvec: np.ndarray, tvec: np.ndarray, noise_px: float,
           rng: np.random.Generator) -> list[tuple[float, float, float, float]]:
    pixels = cv.projectPoints(points_cm, rvec, tvec, intr.K, intr.distortion)[0].reshape(-1, 2)
    pixels = pixels + rng.normal(0.0, noise_px, pixels.shape) if noise_px > 0 else pixels
    width, height = intr.size
    marks = [(0.0, 0.0, 0.0, 0.0)] * 33
    for slot, landmark_id in enumerate(sorted(pose_keypoints)):
        marks[landmark_id] = (float(pixels[slot][0] / width), float(pixels[slot][1] / height), 0.0, 1.0)
    return marks


def write_session(root: str | Path, *, reps: int = 10, lift_cm: float = 13.0, period_s: float = 3.0,
                  still_s: float = 2.0, mac_hz: float = 30.0, pixel_hz: float = 30.0, noise_px: float = 0.5,
                  gaps_s: Iterable[tuple[float, float]] = (), calibration=None,
                  calibration_root: str | Path | None = None, body_mass_kg: float = 65.0,
                  seed: int = 0) -> Path:
    """合成の計測フォルダを ``root`` の下に書き、その場所を返す。gaps_s は Pixel を撮らない時間帯 [s]。"""
    root = Path(root)
    if calibration is None:
        calibration = _default_calibration(Path(calibration_root) if calibration_root else root / "calibration")
    R, T = _extrinsics(calibration)
    rvec1 = cv.Rodrigues(R)[0].ravel()
    profile = dict(lift_cm=lift_cm, period_s=period_s, still_s=still_s)
    seconds = still_s + reps * period_s
    gaps: Sequence[tuple[float, float]] = list(gaps_s)
    events = [("cam0", k / mac_hz) for k in range(int(round(seconds * mac_hz)))]
    events += [("cam1", k / pixel_hz) for k in range(int(round(seconds * pixel_hz)))
               if not any(a <= k / pixel_hz < b for a, b in gaps)]
    recorder = Recorder(calibration, pose_keypoints, root=root, metadata={
        "body_mass_kg": body_mass_kg, "gravity_mode": "axis",
        "synthetic": {"reps": reps, "lift_cm": lift_cm, "period_s": period_s, "still_s": still_s,
                      "mac_hz": mac_hz, "pixel_hz": pixel_hz, "noise_px": noise_px, "gaps_s": gaps,
                      "upper_arm_cm": UPPER_ARM_CM, "forearm_cm": FOREARM_CM, "depth_cm": DEPTH_CM},
    })
    rng = np.random.default_rng(seed)
    seqs = {"cam0": 0, "cam1": 0}
    t0_ns = 1_000_000_000
    for role, t in sorted(events, key=lambda e: (e[1], e[0])):
        body = pushup_body_cm(t, **profile)
        if role == "cam0":
            intr, rvec, tvec = calibration.intrinsics[0], np.zeros(3), np.zeros(3)
        else:
            intr, rvec, tvec = calibration.intrinsics[1], rvec1, T
        width, height = intr.size
        recorder.landmarks(LandmarkFrame(role, seqs[role], t0_ns + round(t * 1e9), width, height,
                                         _marks(body, intr, rvec, tvec, noise_px, rng)))
        seqs[role] += 1
    recorder.close(status="complete", exit_code=0, stop_reason="stop_request")
    return Path(recorder.directory)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="合成の座位プッシュアップを混成の計測フォルダとして書き出す")
    parser.add_argument("--out", required=True, help="計測フォルダを作る親フォルダ")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--lift-cm", type=float, default=13.0)
    parser.add_argument("--period-s", type=float, default=3.0)
    parser.add_argument("--still-s", type=float, default=2.0)
    parser.add_argument("--pixel-hz", type=float, default=30.0)
    parser.add_argument("--noise-px", type=float, default=0.5)
    parser.add_argument("--gap", action="append", default=[], metavar="開始,終わり",
                        help="Pixel を撮らない時間帯 [s]（例 5.0,5.2）。何度でも指定できる")
    parser.add_argument("--calibration", help="投影に使う校正フォルダ（既定は被写体まで 150 cm の合成の配置）")
    parser.add_argument("--body-mass", type=float, default=65.0)
    args = parser.parse_args(argv)
    gaps = [tuple(float(v) for v in text.split(",")) for text in args.gap]
    calibration = load_calibration(args.calibration) if args.calibration else None
    out = write_session(args.out, reps=args.reps, lift_cm=args.lift_cm, period_s=args.period_s,
                        still_s=args.still_s, pixel_hz=args.pixel_hz, noise_px=args.noise_px, gaps_s=gaps,
                        calibration=calibration, body_mass_kg=args.body_mass)
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
