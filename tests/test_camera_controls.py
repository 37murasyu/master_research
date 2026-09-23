"""カメラの設定（露出・WB・AF の固定、解像度など）を録画ツールと計測で共有することを固定する。

**なぜこのテストがあるか。**

録画した映像を計測（``master_research_code.py``）に読み込ませて §6-2・§6-3 を確かめる。録画のときに
計測と違う設定（オート露出など）で撮ると、姿勢推定の雑音が計測と変わり、EKF の較正（S6）が
計測の条件に合わなくなる。そこで計測の ``_apply_camera_controls`` の中身を
``app.core.camera_controls`` に移し、録画ツール（``tools/record_stereo.py``）と共有する。
中身は移しただけで、挙動は変えていない。
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2 as cv
import pytest

from app.core.camera_controls import apply_camera_controls

MAIN_SCRIPT = Path(__file__).resolve().parents[1] / "master_research_code.py"


class FakeCapture:
    def __init__(self, fail=False):
        self.calls = []
        self._fail = fail

    def set(self, prop, value):
        if self._fail:
            raise RuntimeError("このカメラは設定を受け付けない")
        self.calls.append((prop, float(value)))
        return True

    def get(self, prop):
        return 0.0

    def value(self, prop):
        values = [v for p, v in self.calls if p == prop]
        return values[-1] if values else None


class TestDefaults:
    def test_auto_controls_are_turned_off(self):
        cap = FakeCapture()
        apply_camera_controls(cap, 0, env={})
        assert cap.value(cv.CAP_PROP_AUTO_EXPOSURE) == 0.0
        assert cap.value(cv.CAP_PROP_AUTO_WB) == 0.0
        assert cap.value(cv.CAP_PROP_AUTOFOCUS) == 0.0

    def test_nothing_else_is_forced(self):
        cap = FakeCapture()
        apply_camera_controls(cap, 0, env={})
        assert cap.value(cv.CAP_PROP_FRAME_WIDTH) is None
        assert cap.value(cv.CAP_PROP_EXPOSURE) is None

    def test_a_camera_that_refuses_settings_does_not_stop_the_caller(self):
        apply_camera_controls(FakeCapture(fail=True), 0, env={"CAM_WIDTH": "1280", "CAM_EXPOSURE": "-6"})


class TestEnvironment:
    def test_the_per_camera_value_wins(self):
        env = {"CAM_EXPOSURE": "-6", "CAM0_EXPOSURE": "-4"}
        cam0, cam1 = FakeCapture(), FakeCapture()
        apply_camera_controls(cam0, 0, env=env)
        apply_camera_controls(cam1, 1, env=env)
        assert cam0.value(cv.CAP_PROP_EXPOSURE) == -4.0
        assert cam1.value(cv.CAP_PROP_EXPOSURE) == -6.0

    def test_resolution_and_fourcc(self):
        cap = FakeCapture()
        apply_camera_controls(cap, 1, env={"CAM_WIDTH": "1280", "CAM_HEIGHT": "720", "CAM1_FOURCC": "MJPG"})
        assert cap.value(cv.CAP_PROP_FRAME_WIDTH) == 1280.0
        assert cap.value(cv.CAP_PROP_FRAME_HEIGHT) == 720.0
        assert cap.value(cv.CAP_PROP_FOURCC) == float(cv.VideoWriter_fourcc(*"MJPG"))

    @pytest.mark.parametrize("raw, expected", [("on", 1.0), ("1", 1.0), ("off", 0.0)])
    def test_auto_exposure_can_be_left_on(self, raw, expected):
        cap = FakeCapture()
        apply_camera_controls(cap, 0, env={"CAM_AUTO_EXPOSURE": raw})
        assert cap.value(cv.CAP_PROP_AUTO_EXPOSURE) == expected

    def test_without_an_index_only_the_shared_names_apply(self):
        cap = FakeCapture()
        apply_camera_controls(cap, None, env={"CAM_GAIN": "3", "CAM0_GAIN": "9"})
        assert cap.value(cv.CAP_PROP_GAIN) == 3.0


def test_the_measurement_uses_the_shared_controls():
    tree = ast.parse(MAIN_SCRIPT.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "apply_camera_controls"]
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert calls, "計測が共有のカメラ設定を使っていない"
    assert "_set_prop" not in defined, "カメラ設定の本体が計測側に残っている（二重管理になる）"
    for call in calls:
        index = call.args[1] if len(call.args) > 1 else None
        assert index is not None and not (isinstance(index, ast.Constant) and index.value is None), \
            "カメラの番号を渡していない（CAM0_*・CAM1_* が効かなくなる）"
