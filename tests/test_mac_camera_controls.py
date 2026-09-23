"""Mac のカメラに USB と同じカメラ設定を best effort でかけ、結果を残すことを固定する（B9）。

**なぜこのテストがあるか。**

USB の経路は露出・WB・焦点の自動をオフにしてから計測する（``app.core.camera_controls``）。混成の Mac のカメラは
幅と高さしか設定しておらず、自動露出の揺れがそのまま姿勢推定の雑音になっていた。OpenCV の AVFoundation 経路は
これらを設定できない見込みだが（pyobjc は入れない判断）、**何が効いて何が効かなかったかを記録に残さないと**、
朝の実機で設定が効いたのかどうか誰にも分からない。そこで:

- ``apply_camera_controls`` をかけ、``set`` の戻り値と ``get`` の値を ``camera.controls`` に残す（計測の meta に写る）
- 解像度と FOURCC は MacCamera が決める（校正の寸法と合わせる）ので、GUI が渡す ``CAM_WIDTH`` などは使わない
- 起動時の実測 fps と合わせて 1 行で報告する
- 設定を受け付けない・``get`` を持たないカメラでも起動は落とさない
"""

from __future__ import annotations

import itertools

import cv2 as cv
import numpy as np
import pytest

from app.hybrid.mac_camera import MacCamera


class Capture:
    """autofocus だけ受け付け、ほかは拒むカメラ。"""

    def __init__(self):
        self.calls = []
        self.values = {}

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_AUTOFOCUS):
            self.values[prop] = value
            return True
        return False

    def get(self, prop):
        if prop == cv.CAP_PROP_FPS:
            return 30.0
        return self.values.get(prop, -1.0)

    def read(self):
        return True, np.zeros((720, 1280, 3), np.uint8)

    def release(self):
        pass


def _camera(capture, env=None, **kwargs):
    ticks = itertools.count(0.0, 1.0 / 30.0)
    return MacCamera(opener=lambda _: (None, capture), env={} if env is None else env,
                     fps_clock=lambda: next(ticks), **kwargs)


def test_the_results_of_the_controls_are_kept(capsys):
    camera = _camera(Capture())
    controls = camera.controls
    assert controls["applied"]["autofocus"]["ok"] is True
    assert controls["applied"]["autofocus"]["get"] == pytest.approx(0.0)
    assert controls["applied"]["auto_exposure"]["ok"] is False
    assert controls["applied"]["auto_exposure"]["tries"] == 4, "受け付けないと 4 通りの値を試す"
    assert controls["reported_fps"] == pytest.approx(30.0)
    assert controls["measured_fps"] == pytest.approx(30.0, rel=1e-6)
    assert controls["size"] == [1280, 720]
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("[Mac カメラ]")]
    assert len(lines) == 1, out
    assert "autofocus" in lines[0] and "実測" in lines[0]


def test_resolution_and_fourcc_from_the_gui_are_not_applied():
    """GUI は USB 向けの CAM_WIDTH・CAM_FOURCC を渡す。混成の解像度は校正の寸法に合わせる。"""
    capture = Capture()
    _camera(capture, env={"CAM_WIDTH": "640", "CAM0_HEIGHT": "480", "CAM_FOURCC": "MJPG", "CAM1_WIDTH": "320"})
    sizes = [value for prop, value in capture.calls if prop in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT)]
    assert sizes == [1280, 720]
    assert not [call for call in capture.calls if call[0] == cv.CAP_PROP_FOURCC]


def test_fixed_values_from_the_settings_are_tried():
    capture = Capture()
    camera = _camera(capture, env={"CAM0_EXPOSURE": "-6"})
    assert (cv.CAP_PROP_EXPOSURE, -6.0) in capture.calls
    assert camera.controls["applied"]["exposure"]["ok"] is False


def test_a_camera_without_get_still_starts():
    class Plain:
        def set(self, *args):
            pass

        def read(self):
            return True, np.zeros((720, 1280, 3), np.uint8)

        def release(self):
            pass

    camera = _camera(Plain())
    assert camera.controls["applied"]["autofocus"]["ok"] is False
    assert camera.controls["reported_fps"] is None
