"""Exercise the CLI's collection → estimation → automatic save wiring without cameras."""

import itertools
import json
from types import SimpleNamespace
import cv2 as cv
import numpy as np
import pytest
from app.net.protocol import CalibrationFrame, Hello
from app.hybrid.checkerboard import Board
from app.hybrid.calibration_io import load_calibration
from app.runners import hybrid_calibrate as runner
from test_hybrid_calibration import synthetic_views


def _install(monkeypatch, tmp_path, *, collector_errors=0):
    """カメラ・端末・収集器を偽物に差し替える。``collector_errors`` 回ぶん add_remote が失敗する。"""
    from app.hybrid import calibration_io

    board = Board()
    a, b, *_ = synthetic_views(board)
    zero0 = np.zeros((720, 1280), np.uint8)
    zero1 = np.zeros((1080, 1920), np.uint8)
    closed = []
    failures = iter(range(collector_errors))

    class Camera:
        size = (1280, 720)

        def __init__(self, *args):
            pass

        def close(self):
            closed.append("camera")

    class Detector:
        def close(self):
            closed.append("detector")

    class Link:
        url = "ws://127.0.0.1:12345"

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            closed.append("link")

        def take_capture(self):
            return None

        def set_capture_mode(self, mode):
            pass

        def status(self):
            return SimpleNamespace(
                devices={"cam1": Hello("cam1", "Pixel", "s", "pixel-1")}
            )

    class Live:
        local_t_ns = 1
        local_image = cv.cvtColor(zero0, cv.COLOR_GRAY2BGR)
        local_corners = a[0]
        # LiveSession は復号と盤の検出を済ませて置いておく。ランナーはやり直さない
        remote_image = cv.cvtColor(zero1, cv.COLOR_GRAY2BGR)
        remote_corners = b[0]

        def __init__(self, *args):
            pass

        def step(self, **kwargs):
            return CalibrationFrame("cam1", 1, 1, 1920, 1080, b"\xff\xd8\xff\xd9")

    class Collector:
        ready = True
        mono = [a, b]
        sizes = [(1280, 720), (1920, 1080)]
        pairs = list(zip(a, b))
        pair_images = [(zero0, zero1)] * len(a)

        def __init__(self, *args):
            self.cached = (False, False)

        def add_mac(self, *args, **kwargs):
            pass

        def add_remote(self, *args, **kwargs):
            if next(failures, None) is not None:
                raise ValueError("収集中に Pixel の画像寸法が変わりました。やり直してください")

    monkeypatch.setattr(runner, "MacCamera", Camera)
    monkeypatch.setattr(runner, "PoseDetector", Detector)
    monkeypatch.setattr(runner, "PhoneLink", Link)
    monkeypatch.setattr(runner, "LiveSession", Live)
    monkeypatch.setattr(runner, "BoardCollector", Collector)
    monkeypatch.setattr(runner, "poll_window", lambda: 32)
    monkeypatch.setattr(runner, "mac_identity", lambda index: "mac-test")
    # 本物の設定フォルダへ session を書かない
    monkeypatch.setattr(runner, "stable_session", lambda renew=False: "0a1b2c3d")
    monkeypatch.setattr(calibration_io, "calibration_root", lambda: tmp_path)
    monkeypatch.setattr(cv, "imshow", lambda *args: None)
    monkeypatch.setattr(cv, "waitKey", lambda *args: None)
    monkeypatch.setattr(cv, "destroyAllWindows", lambda: None)
    clock = itertools.count(0.0, 2.0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    return closed


def test_runner_estimates_and_saves_then_reuses_cache(tmp_path, monkeypatch):
    closed = _install(monkeypatch, tmp_path)
    for source in ("new", "cache"):
        assert runner.main([]) == 0
        result = load_calibration("latest", root=tmp_path)
        assert all(c["intrinsics_source"] == source for c in result.meta["cameras"])
        assert result.meta["stereo"]["rms"] < 0.01
        assert result.meta["stereo"]["square_error_mm"] < 0.01
    assert closed.count("link") == 2


def test_size_change_during_collection_restarts_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """集めている途中で例外になると、それまでに集めた盤がすべて消える。"""
    _install(monkeypatch, tmp_path, collector_errors=1)
    assert runner.main([]) == 0
    assert "画像寸法が変わりました" in capsys.readouterr().out
    assert load_calibration("latest", root=tmp_path).meta["stereo"]["rms"] < 0.01


def test_failed_estimation_restarts_instead_of_crashing(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, tmp_path)
    real = runner.calibrate_stereo
    calls = iter((True, False))

    def flaky(*args, **kwargs):
        if next(calls, False):
            raise cv.error("縮退した配置")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "calibrate_stereo", flaky)
    assert runner.main([]) == 0
    assert "推定に失敗しました" in capsys.readouterr().out


def test_mismatched_cache_is_not_used(tmp_path, monkeypatch, capsys):
    """カメラ番号が入れ替わる（Camo や iPhone の連係カメラ）と、別のカメラのキャッシュを引きうる。

    以前は警告を出すだけで、合わない K のまま保存していた。
    """
    _install(monkeypatch, tmp_path)
    assert runner.main([]) == 0  # 正しい内部パラメータをキャッシュする

    cached = load_calibration("latest", root=tmp_path).meta["cameras"][0]
    assert cached["kind"] == "mac"
    for path in (tmp_path / "intrinsics").glob("*.json"):
        value = json.loads(path.read_text())
        if value["size"] == [1280, 720]:
            value["K"][0][0] *= 1.3  # 別のカメラの内部パラメータを掴んだ状態
            path.write_text(json.dumps(value))

    assert runner.main([]) == 0
    result = load_calibration("latest", root=tmp_path)
    sources = [c["intrinsics_source"] for c in result.meta["cameras"]]
    assert sources == ["new", "cache"]
    assert "合いません" in capsys.readouterr().out


@pytest.mark.parametrize("value, expected", [("1", 1), ("video=FaceTime", 0), ("", 0)])
def test_camera_default_accepts_any_cam0(monkeypatch, value, expected):
    """USB 経路は CAM0 に動画のパスや "video=..." も許す。混成の起動を落とさない。"""
    from app.hybrid.mac_camera import default_camera_index

    monkeypatch.setenv("CAM0", value)
    assert default_camera_index() == expected
