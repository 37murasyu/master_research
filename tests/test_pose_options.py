"""Mac 側の姿勢推定の設定（``HYBRID_POSE_*``）を固定する（B11）。

**なぜこのテストがあるか。**

USB の経路の姿勢推定の工夫（閾値・縮小・ROI・モデル）を混成でも選べるようにするが、**既定は今と同じ**
（lite・VIDEO モード・閾値 0.5・縮小 1.0・ROI なし）にする。Pixel 側の推定と条件を揃え、VIDEO モードの追跡を
壊さないため。GUI は子プロセスへ設定を全件渡し、その中には USB 向けの既定（``POSE_ROI_ON=1``・
``MP_INPUT_SCALE=0.5``・``POSE_MIN_DET``）がある。混成がそれを読むと、誰も選んでいないのに推定の条件が変わる。
混成は ``HYBRID_POSE_*`` だけを読む。引数なしの ``PoseDetector()`` が環境変数を読むので、``hybrid_measure``・
``hybrid_calibrate``・``hybrid_preview`` は変えずに済む。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.resources import resource_root
from app.core.settings import Settings
from app.hybrid import pose_detector as pd_mod
from app.hybrid.pose_detector import PoseDetector, PoseOptions


def test_the_defaults_are_the_current_behaviour():
    options = PoseOptions()
    assert (options.model, options.min_detection, options.min_presence, options.min_tracking) == (None, 0.5, 0.5, 0.5)
    assert (options.input_scale, options.roi, options.running_mode) == (1.0, False, "VIDEO")


def test_the_usb_defaults_from_the_gui_do_not_leak():
    env = dict(Settings().as_env())
    env.update({"POSE_ROI_ON": "1", "MP_INPUT_SCALE": "0.5", "POSE_MIN_DET": "0.2", "POSE_MIN_TRACK": "0.2"})
    assert PoseOptions.from_env(env) == PoseOptions()


def test_the_hybrid_settings_are_read():
    env = {"HYBRID_POSE_MODEL": "/models/full.task", "HYBRID_POSE_MIN_DET": "0.3", "HYBRID_POSE_MIN_PRESENCE": "0.4",
           "HYBRID_POSE_MIN_TRACK": "0.6", "HYBRID_POSE_INPUT_SCALE": "0.5"}
    options = PoseOptions.from_env(env)
    assert options == PoseOptions(model="/models/full.task", min_detection=0.3, min_presence=0.4, min_tracking=0.6,
                                  input_scale=0.5)


def test_broken_values_fall_back_or_are_clamped():
    options = PoseOptions.from_env({"HYBRID_POSE_MIN_DET": "abc", "HYBRID_POSE_MIN_TRACK": "1.5",
                                    "HYBRID_POSE_INPUT_SCALE": "0.1"})
    assert (options.min_detection, options.min_tracking, options.input_scale) == (0.5, 1.0, 0.25)
    assert PoseOptions.from_env({"HYBRID_POSE_INPUT_SCALE": "3"}).input_scale == 1.0


def test_the_landmarker_gets_the_options():
    built = pd_mod.landmarker_options(PoseOptions(min_detection=0.3, min_presence=0.4, min_tracking=0.6))
    assert built.min_pose_detection_confidence == pytest.approx(0.3)
    assert built.min_pose_presence_confidence == pytest.approx(0.4)
    assert built.min_tracking_confidence == pytest.approx(0.6)
    assert built.running_mode.name == "VIDEO"
    assert built.base_options.model_asset_path == str(resource_root() / "pose_landmarker_lite.task")


def test_a_detector_without_arguments_reads_the_environment(monkeypatch, capsys):
    """``hybrid_measure`` などは ``PoseDetector()`` と書いている。そこで設定が効く。"""
    seen = []

    def create(options):
        seen.append(options)
        return object(), (lambda rgb: rgb)

    monkeypatch.setattr(pd_mod, "_create", create)
    monkeypatch.setenv("HYBRID_POSE_MIN_DET", "0.3")
    monkeypatch.setenv("POSE_MIN_DET", "0.9")
    PoseDetector()
    assert seen[0].min_detection == pytest.approx(0.3)
    assert "[姿勢推定]" in capsys.readouterr().out
    PoseDetector("/models/heavy.task")
    assert seen[1].model == "/models/heavy.task", "位置引数のモデルは設定より優先する"


def test_the_input_is_shrunk_but_the_landmarks_stay_normalised():
    """縮小しても MediaPipe の座標は画像に対する割合なので、全体の座標のまま使える。"""
    shapes = []

    class Point:
        x, y, z, visibility = 0.25, 0.75, 0.0, 0.9

    class Detector:
        def detect_for_video(self, image, stamp):
            shapes.append(image.shape)
            return type("R", (), {"pose_landmarks": [[Point()] * 33]})()

    detector = PoseDetector(detector=Detector(), image_factory=lambda rgb: rgb, options=PoseOptions(input_scale=0.5))
    marks = detector.detect(np.zeros((720, 1280, 3), np.uint8), 1_000_000)
    assert shapes == [(360, 640, 3)]
    assert marks[0] == (0.25, 0.75, 0.0, 0.9)
