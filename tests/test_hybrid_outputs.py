"""混成の計測の記録（``app.hybrid.recorder.Recorder``、計画の T10・T11）。

**なぜこのテストがあるか。**

混成の記録は 3D（EKF なし）・トルク・サイクルの符号付きの仕事しか残しておらず、2026-09-23 の実機の計測では
EKF の較正（``tune_ekf``）にかける生 3D も、回ごとのスコア（W_pos / W_1RM、論文 4.5.2 節）も、関所の状態も
後から確かめられなかった。記録のファイルは足すだけにし（既存の列は変えない）、次を残す:

- ``kpts3d_raw_<stamp>.csv``: EKF の手前の 3D（``RawCaptureWriter`` の形）。1/30 s の格子で、抜けた格子は NaN の行
  （行を詰めると dt 一定の前提が崩れて ``tune_ekf`` の推定が狂う）
- ``cycle_work_<stamp>.csv``: 既存の ``work_j``（符号付き W±）の後ろに W+・W−・W_1RM・スコア
- ``frames_<stamp>.csv``: 既存の列の後ろに格子の番号・dt・関所・高さ・回
- ``meta.json``: 被験者・1RM・体重・前腕長・帯・重力・EKF の出どころ・関所
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from app.gauge.tracker import GaugeTracker
from app.hybrid.measurement import MeasurementSession
from app.net.protocol import LandmarkFrame
from app.runners.network_measure import MeasurementConfig
from app.tuning.raw_capture import read_raw_capture
from config import pose_keypoints
from hybrid_pushup import PushUp
from test_hybrid_measure import calibration

ONE_RM = {"elbow_L": 20.0, "elbow_R": 22.0, "wrist_L": 8.0, "wrist_R": 9.0}
IDS = sorted(pose_keypoints)


def _session(tmp_path, *, reps=1, drop=(), **config):
    """押し上げの合成を本物の MeasurementSession に流した計測フォルダ。校正は test_hybrid_measure の歪みつき。"""
    cal = calibration(tmp_path)
    tracker = GaugeTracker()
    session = MeasurementSession(
        cal, root=tmp_path / "measure", tracker=tracker,
        config=MeasurementConfig(body_mass_kg=65.0, one_rm=ONE_RM, subject_id="00", **config))
    session.on_landmarks(LandmarkFrame("cam1", 0, 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))

    from test_hybrid_measure import geometry
    import cv2 as cv
    from hybrid_pushup import pushup_cm
    from test_network_measure import _pair_from_pixels

    intr, stereo = geometry()
    motion = PushUp(reps=reps)
    pairs = []
    for k in range(int(round(motion.duration_s * 30))):
        if k in drop:
            continue
        truth = pushup_cm(k / 30, motion)
        truth[:, 1] -= 5.0   # 両方の画像に収める
        a = cv.projectPoints(truth, np.zeros(3), np.zeros(3), intr.K, intr.distortion)[0].reshape(-1, 2)
        b = cv.projectPoints(truth, np.zeros(3), stereo.T, intr.K, intr.distortion)[0].reshape(-1, 2)
        pairs.append(_pair_from_pixels(round(k * 1e9 / 30), a, b))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.on_pairs(pairs)
    assert session.exit_code == 0, session.error
    session.stop_reason = "stop_request"
    session.close()
    return session, tracker


def _file(session, prefix):
    stamp = next(session.directory.glob("frames_*.csv")).stem[len("frames_"):]
    return session.directory / f"{prefix}_{stamp}.csv"


class TestCycleWork:
    def test_the_columns_are_appended(self, tmp_path):
        session, _ = _session(tmp_path)
        table = pd.read_csv(_file(session, "cycle_work"))
        assert list(table.columns) == ["frame", "t_ns", "joint", "work_j", "work_pos_j", "work_neg_j", "w1rm_j", "score"]
        elbow = table[table.joint == "elbow_R"].iloc[0]
        assert elbow.work_j == pytest.approx(elbow.work_pos_j + elbow.work_neg_j)
        assert elbow.work_pos_j == pytest.approx(22.0, rel=0.25)
        w1rm = session.measurement.bands["elbow_R"].w1rm
        assert elbow.w1rm_j == pytest.approx(w1rm, rel=1e-6)
        assert elbow.score == pytest.approx(elbow.work_pos_j / w1rm, rel=1e-6)
        shoulder = table[table.joint == "shoulder_R"].iloc[0]
        assert np.isnan(shoulder.w1rm_j) and np.isnan(shoulder.score), "帯が無ければ空"
