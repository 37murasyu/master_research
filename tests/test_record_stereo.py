"""ステレオ録画ツール（``tools/record_stereo.py``）が全フレームを書き、どう止めても動画を閉じることを固定する。

**なぜこのテストがあるか。**

§6-2（ゲージの値・30 fps）、§6-3（EKF の S6 の収録）、§3-2（停止時の CSV）を確かめるには、録画した映像を
計測に読み込ませて処理し直せる必要がある。計測（``master_research_code.py``）自身の録画は、間引きの後の
フレームしか書かないうえ fps 30 固定で書くので、時間が詰まって処理し直せない。そこで録画だけをする
ツールを別に作った。

- 間引かず全フレームを書く。撮影時刻は ``frames.csv`` に残す（容器の fps と実際の fps のずれを見るため）
- 停止ファイル・時間切れ・カメラの失敗・Ctrl-C のどれで止めても、動画を閉じて ``meta.json`` を書く
  （閉じないと mp4/avi が読めなくなる）
- 校正ファイルを録画のフォルダにコピーする。受け取った ``cameras_raw/<試技>/`` と同じ並びになり、
  ``CALIB_BASE_DIR`` にそのまま渡せる
- 解像度が校正と違えば止める（違う解像度の映像を校正の行列で三角測量すると 3D が狂う）
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest

from tools import record_stereo as rs

W, H = 64, 48
CALIB_FILES = ("c0.dat", "c1.dat", "rot_trans_c0.dat", "rot_trans_c1.dat")


class FakeCamera:
    """``n`` 枚まで grab に成功し、その後は失敗する。フレームには番号を焼き込む。"""

    def __init__(self, n: int, fail_retrieve: bool = False):
        self.n = n
        self.count = 0
        self.released = False
        self._fail_retrieve = fail_retrieve

    def grab(self):
        if self.count >= self.n:
            return False
        self.count += 1
        return True

    def retrieve(self):
        if self._fail_retrieve:
            return False, None
        frame = np.full((H, W, 3), (self.count * 7) % 256, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


def _calib_dir(tmp_path):
    calib = tmp_path / "calib"
    calib.mkdir()
    for name in CALIB_FILES:
        (calib / name).write_text(f"{name}\n", encoding="utf-8")
    return calib


def _opener(cameras):
    def open_camera(spec, index, size, allow_size_mismatch):
        return cameras[index], size
    return open_camera


def _run(tmp_path, cameras, *extra):
    argv = ["--cam0", "0", "--cam1", "1", "--calib", str(_calib_dir(tmp_path)), "--out", str(tmp_path / "rec"),
            "--label", "S07", "--width", str(W), "--height", str(H), "--no-preview", *extra]
    code = rs.main(argv, open_camera=_opener(cameras))
    sessions = sorted((tmp_path / "rec").iterdir())
    assert len(sessions) == 1
    return code, sessions[0]


def _frame_count(path):
    cap = cv.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


class TestEveryFrameIsWritten:
    def test_all_grabbed_frames_end_up_in_both_videos(self, tmp_path):
        code, session = _run(tmp_path, [FakeCamera(12), FakeCamera(12)])
        assert code == 0
        videos = sorted(session.glob("cam*_*.avi"))
        assert [v.name.split("_")[0] for v in videos] == ["cam0", "cam1"]
        assert [_frame_count(v) for v in videos] == [12, 12], "間引いたか、動画を閉じずに終わった"

    def test_one_timestamp_row_per_frame(self, tmp_path):
        _, session = _run(tmp_path, [FakeCamera(9), FakeCamera(9)])
        with open(session / "frames.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [int(r["index"]) for r in rows] == list(range(9))
        assert set(rows[0]) == {"index", "t0_ns", "t1_ns", "skew_ms"}

    def test_the_session_can_be_given_to_calib_base_dir(self, tmp_path):
        _, session = _run(tmp_path, [FakeCamera(3), FakeCamera(3)])
        for name in CALIB_FILES:
            assert (session / name).read_text(encoding="utf-8") == f"{name}\n"
        assert session.name.startswith("S07_")


class TestStopping:
    def test_the_stop_file_ends_the_recording_and_the_files_are_closed(self, tmp_path, monkeypatch):
        stop = tmp_path / "stop"
        cameras = [FakeCamera(1000), FakeCamera(1000)]
        original = cameras[1].grab

        def grab_then_request_stop():
            ok = original()
            if cameras[1].count == 5:
                stop.touch()  # 5 枚目を撮ったところで停止を頼む（GUI の停止ボタンと同じ）
            return ok

        cameras[1].grab = grab_then_request_stop
        code, session = _run(tmp_path, cameras, "--stop-file", str(stop))
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert code == 0
        assert meta["stop_reason"] == "stop_request"
        assert meta["frames"] == _frame_count(next(session.glob("cam0_*.avi"))) == 5
        assert all(c.released for c in cameras)

    def test_a_failing_camera_still_closes_everything(self, tmp_path):
        cameras = [FakeCamera(20), FakeCamera(4)]
        code, session = _run(tmp_path, cameras)
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert meta["stop_reason"] == "grab_failed"
        assert meta["frames"] == 4
        assert [_frame_count(v) for v in sorted(session.glob("cam*_*.avi"))] == [4, 4]
        assert code == 0, "カメラが止まっても、それまでの録画は正しく残っている"

    def test_max_frames(self, tmp_path):
        _, session = _run(tmp_path, [FakeCamera(50), FakeCamera(50)], "--max-frames", "7")
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert (meta["stop_reason"], meta["frames"]) == ("max_frames", 7)

    def test_ctrl_c_still_writes_the_meta(self, tmp_path):
        cameras = [FakeCamera(50), FakeCamera(50)]
        original = cameras[0].grab

        def interrupted():
            if cameras[0].count == 3:
                raise KeyboardInterrupt
            return original()

        cameras[0].grab = interrupted
        _, session = _run(tmp_path, cameras)
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert (meta["stop_reason"], meta["frames"]) == ("ctrl_c", 3)
        assert _frame_count(next(session.glob("cam1_*.avi"))) == 3


    def test_an_error_in_the_grab_thread_is_recorded(self, tmp_path):
        """裏のスレッドの例外で record が戻らないと、meta が「0 フレーム」になり録れた分と食い違う。"""
        cameras = [FakeCamera(50), FakeCamera(50)]
        original = cameras[1].grab

        def broken():
            if cameras[1].count == 5:
                raise RuntimeError("USB が抜けた")
            return original()

        cameras[1].grab = broken
        code, session = _run(tmp_path, cameras)
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert meta["frames"] == 5
        assert meta["stop_reason"].startswith("error") and "USB が抜けた" in meta["stop_reason"]
        assert code == 1

    def test_a_leftover_stop_file_does_not_end_the_next_recording(self, tmp_path):
        stop = tmp_path / "stop"
        stop.touch()
        code, session = _run(tmp_path, [FakeCamera(4), FakeCamera(4)], "--stop-file", str(stop))
        assert code == 0
        assert json.loads((session / "meta.json").read_text(encoding="utf-8"))["frames"] == 4


class TestOpeningLikeTheMeasurement:
    """録画は計測（video_io.open_capture_and_read_first）と同じ開き方でカメラを開く。

    Windows では開き方（バックエンド）で露出の値の意味やデバイス番号の並びが変わる。
    """

    class Cap:
        def __init__(self):
            self.sets = []

        def set(self, prop, value):
            self.sets.append((prop, value))
            return True

        def get(self, prop):
            return 0.0

        def read(self):
            return True, np.zeros((H, W, 3), dtype=np.uint8)

        def release(self):
            pass

        def getBackendName(self):
            return "FAKE"

    def test_cameras_get_the_controls_and_files_do_not(self, monkeypatch):
        opened = []

        def opener(src):
            opened.append(src)
            return self.Cap(), True, np.zeros((H, W, 3), dtype=np.uint8)

        monkeypatch.setattr(rs, "open_capture_and_read_first", opener)
        cap, size = rs.open_camera("1", 1, (W, H), False)
        assert opened == [1] and size == (W, H)
        assert (cv.CAP_PROP_FRAME_WIDTH, W) in cap.sets, "解像度を校正に合わせていない"
        file_cap, _ = rs.open_camera("clip.mp4", 0, (W, H), False)
        assert opened[-1] == "clip.mp4" and file_cap.sets == [], "ファイルにカメラの設定を当てた"

    def test_the_backend_is_recorded(self, tmp_path):
        cameras = [FakeCamera(3), FakeCamera(3)]
        for camera in cameras:
            camera.getBackendName = lambda: "MSMF"
        _, session = _run(tmp_path, cameras)
        assert json.loads((session / "meta.json").read_text(encoding="utf-8"))["backends"] == ["MSMF", "MSMF"]


class TestMeta:
    def test_the_meta_describes_the_recording(self, tmp_path):
        _, session = _run(tmp_path, [FakeCamera(6), FakeCamera(6)])
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert meta["frame_size"] == [W, H]
        assert meta["container_fps"] == 30.0
        assert meta["codec"] == "MJPG"
        assert meta["frames"] == 6
        assert meta["calibration_source"].endswith("calib")
        assert "measured_fps" in meta and "max_skew_ms" in meta


def _fake_times(cameras, *, stall_after=None, stall_ms=200.0, skew_ms=1.0):
    """grab の完了時刻を決め打ちにする（30 fps。``stall_after`` 枚目の後で ``stall_ms`` 止まる。cam1 は ``skew_ms`` 遅れる）。"""
    def timed_grab(camera, clock):
        ok = camera.grab()
        k = camera.count
        t_ms = k * 1000.0 / 30.0 + (stall_ms if stall_after is not None and k > stall_after else 0.0)
        t_ms += skew_ms if camera is cameras[1] else 0.0
        return ok, int(round(t_ms * 1e6))
    return timed_grab


class TestGapsAndSkew:
    """USB の取りこぼしで録画が 0.2 s 止まっても、以前は何も言わなかった（frames.csv には間隔が残るが、誰も読まない）。
    再生は動画のフレームを 1 フレームの間隔で並べ直すので、その区間の時間が詰まる。左右の撮影時刻のずれも、再生は
    同じ番号のフレームを組にするので三角測量に効く。frames.csv から穴とずれを数えて meta に残し、警告する。"""

    def test_a_stall_is_counted_and_warned(self, tmp_path, monkeypatch, capsys):
        cameras = [FakeCamera(12), FakeCamera(12)]
        monkeypatch.setattr(rs, "_timed_grab", _fake_times(cameras, stall_after=6))
        code, session = _run(tmp_path, cameras)
        timing = json.loads((session / "meta.json").read_text(encoding="utf-8"))["frame_timing"]
        assert code == 0
        assert timing["gaps"] == 1 and timing["missing_frames"] == 6
        assert timing["max_interval_ms"] == pytest.approx(1000 / 30 + 200, abs=0.01)
        assert timing["median_interval_ms"] == pytest.approx(1000 / 30, abs=0.01)
        out = capsys.readouterr().out
        assert "[WARN]" in out and "穴が 1 か所" in out

    def test_left_right_skew_is_counted_and_warned(self, tmp_path, monkeypatch, capsys):
        cameras = [FakeCamera(9), FakeCamera(9)]
        monkeypatch.setattr(rs, "_timed_grab", _fake_times(cameras, skew_ms=20.0))
        _, session = _run(tmp_path, cameras)
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert meta["frame_timing"]["skewed_frames"] == 9 and meta["frame_timing"]["gaps"] == 0
        assert meta["max_skew_ms"] == pytest.approx(20.0)
        assert "ずれ" in capsys.readouterr().out

    def test_a_steady_recording_is_not_warned(self, tmp_path, monkeypatch, capsys):
        cameras = [FakeCamera(9), FakeCamera(9)]
        monkeypatch.setattr(rs, "_timed_grab", _fake_times(cameras))
        _, session = _run(tmp_path, cameras)
        timing = json.loads((session / "meta.json").read_text(encoding="utf-8"))["frame_timing"]
        assert (timing["gaps"], timing["skewed_frames"]) == (0, 0)
        assert "[WARN]" not in capsys.readouterr().out

    def test_the_timing_can_be_read_back_from_frames_csv(self, tmp_path, monkeypatch):
        """古い録画（meta に frame_timing の無いもの）も、再生の報告で frames.csv から数え直せる。"""
        cameras = [FakeCamera(12), FakeCamera(12)]
        monkeypatch.setattr(rs, "_timed_grab", _fake_times(cameras, stall_after=6))
        _, session = _run(tmp_path, cameras)
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        assert rs.read_frame_timing(session) == pytest.approx(meta["frame_timing"])


class TestFrameSize:
    def test_a_mismatch_with_the_calibration_stops(self):
        with pytest.raises(rs.FrameSizeMismatch, match="1280x720"):
            rs.check_frame_size((640, 480), (1280, 720), allow=False)

    def test_a_mismatch_can_be_allowed(self):
        rs.check_frame_size((640, 480), (1280, 720), allow=True)

    def test_the_default_size_and_cameras_come_from_the_calibration_settings(self, tmp_path):
        settings = tmp_path / "calibration_settings.yaml"
        settings.write_text("camera0: 2\ncamera1: 3\nframe_width: 1280\nframe_height: 720\n", encoding="utf-8")
        assert rs.calibration_defaults(settings) == SimpleNamespace(cam0="2", cam1="3", width=1280, height=720)


def test_missing_calibration_files_are_reported_before_opening_cameras(tmp_path):
    calib = tmp_path / "calib"
    calib.mkdir()
    opened = []

    def open_camera(*args):
        opened.append(args)
        raise AssertionError("校正ファイルが無いのにカメラを開いた")

    code = rs.main(["--calib", str(calib), "--out", str(tmp_path / "rec"), "--no-preview"], open_camera=open_camera)
    assert code == 2
    assert not opened
