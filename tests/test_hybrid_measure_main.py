"""混成の計測の子プロセス ``app.runners.hybrid_measure.main`` のメインループを通す（計画の T12）。

**なぜこのテストがあるか。**

``hybrid_measure.main`` のループ（Mac の取得・推定・表示を回し、停止の要求で抜けて記録を閉じる）を通すテストは
これまで無かった。ゲージ（GUI の Qt 窓）は子の標準出力の ``@@GAUGE {json}`` の行だけを見て動くので、
メインループの 1 周（約 30 Hz）ごとに 1 行、``sys.stdout.write`` の 1 回で出ていること、行が v2 の形で読めること、
Pixel の点が届いたら ``link`` が connected になり、押し上げで ``rep`` が増えることを、カメラ・姿勢推定・PhoneLink を
偽物に替えて確かめる。設定（被験者番号・1RM の表・関所・EKF）も環境変数から読まれることを見る。
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from app.gauge import protocol
from app.hybrid.calibration_io import update_meta
from app.net.protocol import LandmarkFrame
from app.runners import hybrid_measure
from hybrid_pushup import PushUp, calibrated_pairs
from test_hybrid_measure import calibration

ONE_RM_TABLE = "subject_id,elbow_L_outer,elbow_R_outer,wrist_L,wrist_R\n0,20,22,8,9\n"


def _mac_identity(index):
    """校正ランナーと同じ形（``calibration_io.mac_identity``、``<機種>:camera<番号>``）の Mac のカメラの識別子。"""
    return f"Mac14,2:camera{index}"


def _set_mac_camera(cal, device_id):
    """校正の meta の Mac のカメラの識別子（``cameras[0].device_id``）を書き換える。None なら鍵を消す（古い校正）。"""
    cameras = [dict(camera) for camera in cal.meta["cameras"]]
    cameras[0].pop("device_id", None)
    if device_id is not None:
        cameras[0]["device_id"] = device_id
    update_meta(cal.directory, cameras=cameras)


class FakeCamera:
    def __init__(self, index, size=None):
        self.index, self.size = index, size

    def close(self):
        pass


class FakeDetector:
    def close(self):
        pass


class FakeLink:
    """PhoneLink の代わり。受信スレッドの呼び出し（on_landmarks・on_pairs）は FakeLive.step がその場で行う。"""

    instances: list["FakeLink"] = []

    def __init__(self, **kwargs):
        self.callbacks = kwargs
        self.url = "ws://fake"
        self.remote_role = "cam1"
        self.connected = True
        FakeLink.instances.append(self)

    def start(self):
        pass

    def stop(self):
        self.callbacks["on_stop"]()

    def status(self):
        return SimpleNamespace(devices={"cam1": "pixel-1"} if self.connected else {})


class FakeLive:
    """1 周ごとに Pixel の点を 1 つと押し上げの組を 3 組ずつ流す（Pixel 側の受信を模す）。流し終えたら停止を要求する。

    ``disconnect`` なら、最後の組を流したところで Pixel の接続を切る。
    """

    pairs: list = []
    disconnect = False

    def __init__(self, camera, detector, link, **kwargs):
        self.link = link
        self.steps = 0
        self.index = 0

    def step(self, **kwargs):
        cb = self.link.callbacks
        if self.steps >= 2:
            cb["on_landmarks"](LandmarkFrame("cam1", self.steps, self.steps, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))
            chunk = FakeLive.pairs[self.index:self.index + 3]
            self.index += 3
            if chunk:
                cb["on_pairs"](chunk)
            if FakeLive.disconnect and self.index >= len(FakeLive.pairs):
                self.link.connected = False
            cb["on_tick"]()
        self.steps += 1
        FakeStop.done = self.index >= len(FakeLive.pairs)


class FakeStop:
    done = False

    @classmethod
    def from_environment(cls):
        return cls()

    def install_signal_handlers(self):
        pass

    def requested(self):
        return FakeStop.done


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    FakeLink.instances.clear()
    FakeStop.done = False
    FakeLive.pairs = calibrated_pairs(PushUp(reps=2))
    FakeLive.disconnect = False
    monkeypatch.setattr(hybrid_measure, "MacCamera", FakeCamera)
    monkeypatch.setattr(hybrid_measure, "PoseDetector", FakeDetector)
    monkeypatch.setattr(hybrid_measure, "PhoneLink", FakeLink)
    monkeypatch.setattr(hybrid_measure, "LiveSession", FakeLive)
    monkeypatch.setattr(hybrid_measure, "poll_window", lambda: -1)
    monkeypatch.setattr(hybrid_measure, "StopRequest", FakeStop)
    monkeypatch.setattr(hybrid_measure, "stable_session", lambda renew=False: "abcd1234")
    monkeypatch.setattr(hybrid_measure, "mac_identity", _mac_identity)
    monkeypatch.setattr("app.hybrid.recorder.measurement_root", lambda: tmp_path / "measure")
    table = tmp_path / "one_rm.csv"
    table.write_text(ONE_RM_TABLE, encoding="utf-8")
    monkeypatch.setenv("SUBJECT_ID", "00")
    monkeypatch.setenv("ONE_RM_CSV", str(table))
    monkeypatch.setenv("BODY_MASS_KG", "65")
    monkeypatch.delenv("HYBRID_DYN_GATE", raising=False)
    monkeypatch.delenv("HYBRID_EKF_PROFILE", raising=False)
    monkeypatch.delenv("DEMO_MONO_GAUGE_ON", raising=False)
    monkeypatch.delenv("CAM0", raising=False)
    cal = calibration(tmp_path)
    _set_mac_camera(cal, _mac_identity(0))  # 校正は既定の 0 番のカメラで取った
    return cal


def _gauge_lines(out: str) -> list[str]:
    return [line + "\n" for line in out.splitlines() if line.startswith(protocol.PREFIX)]


def test_every_loop_writes_one_gauge_line(fakes, capsys):
    code = hybrid_measure.main(["--calibration", str(fakes.directory)])
    out = capsys.readouterr().out
    assert code == 0
    lines = _gauge_lines(out)
    frames = [protocol.decode(line) for line in lines]
    assert all(frame is not None for frame in frames), "v2 の形で読めない行がある"
    assert all(len(line.encode("utf-8")) < 512 for line in lines)
    # 1 周 1 行（間引かない）＋終わりの 1 行
    assert len(lines) >= len(FakeLive.pairs) // 3 + 1
    assert frames[0].link == "waiting" and frames[-1].link == "connected"
    assert frames[-1].source == "measure"
    assert frames[-1].rep == 2, "押し上げ 2 回で rep が 2"
    elbow = frames[-1].parts["elbow_R"]
    assert elbow.band is not None and elbow.band[0] < elbow.band[1]
    assert elbow.prev is not None and elbow.prev > 5.0
    assert max(f.parts["elbow_R"].now or 0.0 for f in frames) > 5.0, "押し上げの途中で now が増える"
    assert set(frames[-1].parts) == set(protocol.PART_NAMES)


def test_the_gauge_line_says_waiting_after_the_pixel_disconnects(fakes, capsys):
    """Pixel の接続が切れたら、次の定期処理（on_tick）でゲージの行の link が waiting に戻る（GUI の接続表示）。

    以前は記録を始めたときに 1 回 connected にするだけで、切れても connected のままだった。
    """
    FakeLive.disconnect = True
    assert hybrid_measure.main(["--calibration", str(fakes.directory)]) == 0
    frames = [protocol.decode(line) for line in _gauge_lines(capsys.readouterr().out)]
    assert any(frame.link == "connected" for frame in frames)
    assert frames[-1].link == "waiting"


def test_the_settings_reach_the_measurement(fakes, capsys, monkeypatch):
    monkeypatch.setenv("HYBRID_DYN_GATE", "0")
    monkeypatch.setenv("EKF_ENABLE", "0")
    assert hybrid_measure.main(["--calibration", str(fakes.directory)]) == 0
    out = capsys.readouterr().out
    folder = next(line.split("保存: ", 1)[1] for line in out.splitlines() if line.startswith("保存: "))
    meta = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    assert meta["subject_id"] == "00"
    assert meta["one_rm_kg"]["elbow_R"] == 22.0
    assert meta["dyn_gate"] is False
    assert meta["ekf"]["enabled"] is False


def test_a_missing_one_rm_table_gives_no_band(fakes, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("ONE_RM_CSV", str(tmp_path / "missing.csv"))
    assert hybrid_measure.main(["--calibration", str(fakes.directory)]) == 0
    captured = capsys.readouterr()
    last = protocol.decode(_gauge_lines(captured.out)[-1])
    assert all(reading.band is None for reading in last.parts.values())
    assert "1RM" in captured.err


def test_the_demo_moves_the_gauge_without_torque(fakes, capsys, monkeypatch):
    """``DEMO_MONO_GAUGE_ON=1``（GUI の既定は 0）なら、3D の肩の上昇と肘角の変化で針を動かし、行の source を demo にする。"""
    monkeypatch.setenv("DEMO_MONO_GAUGE_ON", "1")
    assert hybrid_measure.main(["--calibration", str(fakes.directory)]) == 0
    captured = capsys.readouterr()
    frames = [protocol.decode(line) for line in _gauge_lines(captured.out)]
    assert all(frame.source == "demo" for frame in frames)
    assert max(f.parts["elbow_R"].now or 0.0 for f in frames) > 1.0
    folder = next(line.split("保存: ", 1)[1] for line in captured.out.splitlines() if line.startswith("保存: "))
    assert json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))["demo"] is True


def _recording_camera(monkeypatch):
    """開いたカメラの番号を残す MacCamera の偽物。"""
    opened = []

    def camera(index, size=None):
        opened.append(index)
        return FakeCamera(index, size)

    monkeypatch.setattr(hybrid_measure, "MacCamera", camera)
    return opened


def test_a_mac_camera_other_than_the_calibrated_one_stops(fakes, capsys, monkeypatch, tmp_path):
    """校正は 1 番のカメラ、計測は 0 番なら、カメラを開かずに終了コード 2 で止め、両方の識別子を理由に出す。

    以前は Pixel の端末 ID しか見なかった。Camo や iPhone の連係カメラで番号がずれたまま計測すると、別のカメラの
    画像に校正を当て、3D とトルクが丸ごと狂ったまま complete の記録が残った。止めるのは記録を始める前（記録は
    Pixel の最初の点で始まる）なので、計測フォルダは作らない。
    """
    _set_mac_camera(fakes, _mac_identity(1))
    opened = _recording_camera(monkeypatch)
    code = hybrid_measure.main(["--calibration", str(fakes.directory), "--camera", "0"])
    err = capsys.readouterr().err
    assert code == 2
    assert opened == [], "校正と違うカメラを開いた"
    assert _mac_identity(1) in err and _mac_identity(0) in err
    assert not (tmp_path / "measure").exists(), "止めたのに計測フォルダができた"


def test_the_calibrated_mac_camera_is_measured(fakes, capsys, monkeypatch):
    """校正と同じ番号（設定 CAM0 から読む番号）なら、そのまま計測する。"""
    _set_mac_camera(fakes, _mac_identity(1))
    monkeypatch.setenv("CAM0", "1")
    opened = _recording_camera(monkeypatch)
    assert hybrid_measure.main(["--calibration", str(fakes.directory)]) == 0
    assert opened == [1]
    assert "保存: " in capsys.readouterr().out


def _unidentifiable(index):
    raise subprocess.CalledProcessError(1, ["sysctl", "-n", "hw.model"])


@pytest.mark.parametrize("case", ["old_calibration", "no_identity"])
def test_an_unverifiable_mac_camera_only_warns(fakes, capsys, monkeypatch, case):
    """meta に識別子が無い古い校正と、今のカメラの識別子を取れない（sysctl の失敗）ときは、警告だけで計測する。"""
    if case == "old_calibration":
        _set_mac_camera(fakes, None)
    else:
        monkeypatch.setattr(hybrid_measure, "mac_identity", _unidentifiable)
    opened = _recording_camera(monkeypatch)
    assert hybrid_measure.main(["--calibration", str(fakes.directory)]) == 0
    captured = capsys.readouterr()
    assert opened == [0]
    assert "保存: " in captured.out
    assert "確かめられない" in captured.err


def test_hybrid_measure_ignores_hybrid_replay(monkeypatch):
    """``hybrid_measure`` は常に実機の計測。``HYBRID_REPLAY`` があっても再生へ回さない。

    以前は環境変数を見て再生へ回していたので、親のシェルに残った ``export HYBRID_REPLAY=...`` で本番の計測が
    黙って再生になった。再生は独立の role（``hybrid_replay``）で、GUI は「記録の再生」の入力で選ぶ。
    """
    import app.runners.hybrid_replay as hybrid_replay

    def replay_main(argv=None):
        raise AssertionError("HYBRID_REPLAY を見て再生へ回した")

    def load_calibration(name):
        raise ValueError("校正が無い（実機の道を通った印）")

    monkeypatch.setenv("HYBRID_REPLAY", "/somewhere/measure/20260923_000000_000000")
    monkeypatch.setattr(hybrid_replay, "main", replay_main)
    monkeypatch.setattr(hybrid_measure, "load_calibration", load_calibration)
    assert hybrid_measure.main([]) == 2
    source = Path(hybrid_measure.__file__).read_text(encoding="utf-8")
    assert "HYBRID_REPLAY" not in source and "hybrid_replay" not in source
