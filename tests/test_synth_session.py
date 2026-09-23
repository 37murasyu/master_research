"""合成の座位プッシュアップを混成の計測フォルダとして書き出す道具（``tools.synth_session``）。

**なぜこのテストがあるか。**

2026-09-23 の実機の記録は、上下の動きが 1〜3 cm しかない区間が多く（置き方の失敗で右腕の 3D も飛ぶ）、
「押し上げ 1 回ごとに回が閉じ、論文の閾値を跨ぐとゲージの状態が変わる」ことの根拠にならない。
そこで、手を肘掛けに固定し、肩と腰が決まった高さだけ持ち上がる体を 2 台のカメラへ投影し、本物の Recorder で
``landmarks2d_*`` を書く。再生（``app.hybrid.replay``）がこれを本番と同じ道筋で計測に通す。
体の形（上腕 30 cm・前腕 25 cm で長さが保たれる、手が動かない、持ち上げの高さ）が崩れると検証の前提が崩れる。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tools.synth_session as ss
from app.hybrid.retriangulate import read_landmarks
from config import pose_keypoints, slot_of


class TestBody:
    def test_the_hands_stay_on_the_armrests(self):
        a = ss.pushup_body_cm(0.0)
        b = ss.pushup_body_cm(3.8)   # 持ち上げの途中
        for name in ("L_WRIST", "R_WRIST"):
            assert np.allclose(a[slot_of(name)], b[slot_of(name)])

    @pytest.mark.parametrize("t", [0.0, 2.5, 3.2, 3.9, 5.0])
    def test_the_arm_segments_keep_their_length(self, t):
        body = ss.pushup_body_cm(t)
        for side in "LR":
            upper = np.linalg.norm(body[slot_of(f"{side}_ELBOW")] - body[slot_of(f"{side}_SHOULDER")])
            fore = np.linalg.norm(body[slot_of(f"{side}_WRIST")] - body[slot_of(f"{side}_ELBOW")])
            assert upper == pytest.approx(ss.UPPER_ARM_CM, abs=1e-6)
            assert fore == pytest.approx(ss.FOREARM_CM, abs=1e-6)

    def test_the_shoulders_rise_by_the_lift_and_come_back(self):
        """カメラ座標の y は下向きなので、持ち上がると y が減る。"""
        lift, period, still = 13.0, 3.0, 2.0
        heights = [-np.mean(ss.pushup_body_cm(t, lift_cm=lift, period_s=period, still_s=still)[
            [slot_of("L_SHOULDER"), slot_of("R_SHOULDER")], 1]) for t in np.arange(0, still + period, 0.01)]
        base = heights[0]
        assert max(heights) - base == pytest.approx(lift, abs=0.05)
        assert heights[-1] == pytest.approx(base, abs=0.05), "1 回の終わりに座った高さへ戻っていない"
        assert max(abs(h - base) for h in heights[: int(still / 0.01)]) < 1e-9, "最初の静止の間に動いている"


class TestSession:
    def test_the_session_can_be_read_back(self, tmp_path):
        out = ss.write_session(tmp_path / "measure", reps=2, period_s=3.0, still_s=2.0, pixel_hz=15.0,
                               calibration_root=tmp_path / "calibration")
        frames = read_landmarks(out)
        seconds = 2.0 + 2 * 3.0
        assert len(frames["cam0"]) == pytest.approx(30 * seconds, abs=2)
        assert len(frames["cam1"]) == pytest.approx(15 * seconds, abs=2)
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["synthetic"]["reps"] == 2
        assert meta["status"] == "complete"

    def test_every_point_is_inside_both_images(self, tmp_path):
        out = ss.write_session(tmp_path / "measure", reps=1, calibration_root=tmp_path / "calibration")
        frames = read_landmarks(out)
        for role in ("cam0", "cam1"):
            for frame in frames[role]:
                for landmark_id in pose_keypoints:
                    x, y, _, visibility = frame.landmarks[landmark_id]
                    assert 0.0 < x < 1.0 and 0.0 < y < 1.0, f"{role} の {landmark_id} が画面の外"
                    assert visibility == 1.0

    def test_pixel_gaps_leave_holes(self, tmp_path):
        """組の抜け（100 ms 超）を再現できる。dt の修正と EKF の予測を試すため。"""
        out = ss.write_session(tmp_path / "measure", reps=1, gaps_s=[(2.5, 2.7)],
                               calibration_root=tmp_path / "calibration")
        times = np.array([f.t_capture_ns for f in read_landmarks(out)["cam1"]]) / 1e9
        times -= times[0]
        assert not np.any((times > 2.5) & (times < 2.7))
