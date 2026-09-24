"""Mac 側の姿勢推定の設定（``HYBRID_POSE_*``）を固定する（B11）。

**なぜこのテストがあるか。**

USB の経路の姿勢推定の工夫（閾値・縮小・ROI・モデル）を混成でも選べるようにするが、**既定は今と同じ**
（lite・VIDEO モード・閾値 0.5・縮小 1.0・ROI なし）にする。Pixel 側の推定と条件を揃え、VIDEO モードの追跡を
壊さないため。GUI は子プロセスへ設定を全件渡し、その中には USB 向けの既定（``POSE_ROI_ON=1``・
``MP_INPUT_SCALE=0.5``・``POSE_MIN_DET``）がある。混成がそれを読むと、誰も選んでいないのに推定の条件が変わる。
混成は ``HYBRID_POSE_*`` だけを読む。引数なしの ``PoseDetector()`` が環境変数を読むので、``hybrid_measure``・
``hybrid_calibrate``・``hybrid_preview`` は変えずに済む。

``HYBRID_POSE_ROI=1`` の配線（IMAGE モード、前のフレームの点から ROI を切り出す、点を全体の座標へ戻す、見失ったら
広げて全画面へ戻す）は、MediaPipe の推定器を偽物に差し替えて確かめる。式そのものは ``test_pose_roi.py``。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.resources import resource_root
from app.core.settings import Settings
from app.hybrid import pose_detector as pd_mod
from app.hybrid import pose_roi
from app.hybrid.pose_detector import PoseDetector, PoseOptions
from config import pose_keypoints


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


# --- ROI（HYBRID_POSE_ROI=1）の配線 ------------------------------------------------------
#
# ROI が有効なら IMAGE モードで推定し、前のフレームの点から決めた ROI を切り出して渡し、返る点を全体の正規化座標へ
# 戻す。点を見失ったら本体と同じく ROI を広げ、POSE_ROI_MAX_MISS（4）回を超えたら全画面に戻す。


FULL = (720, 1280, 3)


class _Mark:
    def __init__(self, x, y, z=0.0, visibility=0.9):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def _body(x=0.4, y=0.4):
    """pose_keypoints が (x..x+0.05, y..y+0.15) に散らばる 33 点。"""
    marks = [_Mark(0.5, 0.5) for _ in range(33)]
    for k, i in enumerate(sorted(pose_keypoints)):
        marks[i] = _Mark(x + 0.05 * (k % 2), y + 0.15 * (k // 8))
    return marks


class _ImageDetector:
    """IMAGE モードの偽の推定器。渡された画像を覚え、決めた順に結果を返す。"""

    def __init__(self, answers):
        self.images, self.answers = [], list(answers)

    def detect(self, image):
        self.images.append(image)
        marks = self.answers.pop(0) if self.answers else None
        return type("R", (), {"pose_landmarks": [marks] if marks else []})()

    def detect_for_video(self, image, stamp):
        raise AssertionError("ROI が有効なら IMAGE モード（detect）で推定する")


def _gradient():
    """画素ごとに値の違う画像（切り出した場所を中身で確かめるため）。"""
    ys, xs = np.mgrid[0:FULL[0], 0:FULL[1]]
    return np.dstack([xs % 251, ys % 241, (xs + ys) % 239]).astype(np.uint8)


def _expected_roi(marks):
    points = [(m.x, m.y, m.z, m.visibility) for m in marks]
    return pose_roi.roi_from_keypoints(pose_roi.landmarks_to_pixels(points, FULL, pose_keypoints), FULL)


def test_the_roi_option_builds_an_image_mode_landmarker_and_says_so(monkeypatch, capsys):
    seen = []

    def create(options):
        seen.append(options)
        return object(), (lambda rgb: rgb)

    monkeypatch.setattr(pd_mod, "_create", create)
    monkeypatch.setenv("HYBRID_POSE_ROI", "1")
    detector = PoseDetector()
    assert seen[0].roi and seen[0].running_mode == "IMAGE" and detector.options.roi
    out = capsys.readouterr().out
    assert "IMAGE モード" in out and "ROI あり" in out and "未配線" not in out
    assert pd_mod.landmarker_options(PoseOptions(roi=True)).running_mode.name == "IMAGE"


def test_the_roi_crops_the_image_and_the_landmarks_come_back_to_the_full_frame():
    body = _body()
    fake = _ImageDetector([body, [_Mark(0.5, 0.25, -0.2, 0.7)] * 33])
    detector = PoseDetector(detector=fake, image_factory=lambda rgb: rgb, options=PoseOptions(roi=True))
    image = _gradient()

    first = detector.detect(image, 1_000_000)
    assert fake.images[0].shape == FULL, "最初は前の点が無いので全画面"
    assert first[0] == (0.5, 0.5, 0.0, 0.9), "全画面の点はそのまま"

    x0, y0, x1, y1 = _expected_roi(body)
    assert (x1 - x0, y1 - y0) != (1280, 720)
    second = detector.detect(image, 2_000_000)
    assert np.array_equal(fake.images[1], image[y0:y1, x0:x1, ::-1]), "前のフレームの点から決めた ROI を RGB で渡す"
    assert second[0] == pytest.approx(((x0 + 0.5 * (x1 - x0)) / 1280, (y0 + 0.25 * (y1 - y0)) / 720, -0.2, 0.7))
    assert len(second) == 33


def test_the_roi_is_shrunk_like_the_full_frame():
    body = _body()
    fake = _ImageDetector([body, body])
    detector = PoseDetector(detector=fake, image_factory=lambda rgb: rgb,
                            options=PoseOptions(roi=True, input_scale=0.5))
    detector.detect(_gradient(), 1_000_000)
    detector.detect(_gradient(), 2_000_000)
    x0, y0, x1, y1 = _expected_roi(body)
    assert fake.images[0].shape == (360, 640, 3)
    assert fake.images[1].shape == (round((y1 - y0) * 0.5), round((x1 - x0) * 0.5), 3)


def test_lost_landmarks_grow_the_roi_and_then_fall_back_to_the_full_frame():
    body = _body()
    fake = _ImageDetector([body] + [None] * 7)
    detector = PoseDetector(detector=fake, image_factory=lambda rgb: rgb, options=PoseOptions(roi=True))
    image = np.zeros(FULL, np.uint8)
    marks = [detector.detect(image, (k + 1) * 1_000_000) for k in range(8)]
    assert marks[1:] == [None] * 7, "見失ったフレームは点なし（呼ぶ側から見た形は今と同じ）"

    roi = _expected_roi(body)
    expected = [FULL[:2]]
    for miss in range(pose_roi.MAX_MISS + 1):
        grown = roi
        for _ in range(miss):
            grown = pose_roi.expand_roi(grown, FULL)
        expected.append((grown[3] - grown[1], grown[2] - grown[0]))
    expected += [FULL[:2]] * (8 - len(expected))
    assert [im.shape[:2] for im in fake.images] == expected, "本体と同じく見失った回数だけ広げ、4 回を超えたら全画面"


def test_too_few_points_keep_the_full_frame():
    few = [_Mark(-0.1, -0.1) for _ in range(33)]
    for i in sorted(pose_keypoints)[:3]:
        few[i] = _Mark(0.5, 0.5)
    fake = _ImageDetector([few, few])
    detector = PoseDetector(detector=fake, image_factory=lambda rgb: rgb, options=PoseOptions(roi=True))
    detector.detect(np.zeros(FULL, np.uint8), 1_000_000)
    detector.detect(np.zeros(FULL, np.uint8), 2_000_000)
    assert [im.shape for im in fake.images] == [FULL, FULL]


def test_without_the_roi_the_detector_is_called_as_before():
    """ROI が無効なら毎回全画面を VIDEO モード（detect_for_video）で、時刻は単調に増える ms。"""
    calls = []

    class Detector:
        def detect_for_video(self, image, stamp):
            calls.append((image.shape, stamp))
            return type("R", (), {"pose_landmarks": [_body()]})()

    detector = PoseDetector(detector=Detector(), image_factory=lambda rgb: rgb, options=PoseOptions())
    for t_ns in (5_000_000, 5_000_000, 7_000_000):
        detector.detect(np.zeros(FULL, np.uint8), t_ns)
    assert calls == [(FULL, 5), (FULL, 6), (FULL, 7)]
