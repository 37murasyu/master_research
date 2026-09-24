"""``pose_runtime.PoseEstimator`` が、推論の失敗した 1 フレームで落ちないことを固定する。

**なぜこのテストがあるか。**

native（DLL）の経路で 1 フレームの推論が失敗すると、``process`` は tasks の経路へ進めず（mode が native のまま）、
最後の ``self._pose.process`` に落ちた。native では Solutions の Pose を作っていない（``_pose`` が None）ので
``AttributeError`` になり、計測が止まった。tasks の経路は失敗したらその場で Solutions の Pose を作って代わりに
使うので、native も同じ扱いにする。
"""

from __future__ import annotations

import numpy as np
import pytest

pr = pytest.importorskip("pose_runtime")   # mediapipe が要る

FRAME = np.zeros((10, 10, 3), np.uint8)


class FakeSolutionsPose:
    def __init__(self, **options):
        self.options = options
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        return "solutions"


class Failing:
    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        raise RuntimeError("推論が 1 フレーム失敗")

    detect = process


def _estimator(mode, **parts):
    est = pr.PoseEstimator.__new__(pr.PoseEstimator)
    est._mode, est._native, est._pose, est._landmarker = mode, parts.get("native"), None, parts.get("landmarker")
    return est


@pytest.fixture
def solutions(monkeypatch):
    made = []

    def make(**options):
        made.append(FakeSolutionsPose(**options))
        return made[-1]

    monkeypatch.setattr(pr, "_solutions_pose", make)
    return made


def test_a_failed_native_frame_falls_back_to_solutions(solutions):
    native = Failing()
    est = _estimator("native", native=native)
    assert est.process(FRAME) == "solutions"
    assert len(solutions) == 1 and est._mode == "solutions"
    assert est.process(FRAME) == "solutions" and native.calls == 1, "切り替えた後も失敗する DLL を呼んでいる"
    assert len(solutions) == 1, "フレームごとに Solutions の Pose を作り直している"


def test_a_failed_tasks_frame_falls_back_the_same_way(solutions, monkeypatch):
    monkeypatch.setattr(pr.mp, "Image", lambda **kwargs: None)
    est = _estimator("tasks", landmarker=Failing())
    assert est.process(FRAME) == "solutions"
    assert est._mode == "solutions" and len(solutions) == 1


def test_no_solutions_either_gives_an_empty_result(monkeypatch):
    def broken(**options):
        raise RuntimeError("Solutions も無い")

    monkeypatch.setattr(pr, "_solutions_pose", broken)
    result = _estimator("native", native=Failing()).process(FRAME)
    assert result.pose_landmarks is None
