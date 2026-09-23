"""計測画面（``app.shell.page_measure.MeasurePage``）の試験。設計書 §5.2。

見た目は検証しない。確かめるのは、主ボタン 1 つでの開始と停止、「J の数値」スイッチ、
被験者ゲージの窓（``app.gauge.window.GaugeWindow``）との結線。

子プロセスを本当に起動するのは、主ボタンで開始して止める試験と、起動に失敗する試験だけ。
ほかは ``runner.start`` を差し替え、フレームや終了は ``runner`` のシグナルを直接出して確かめる
（controller の「判断済みのこと」）。ゲージ窓は offscreen では画面が 1 つなので 1280×720 の窓になる。
"""

from __future__ import annotations

import sys

import pytest

from app.core.settings import Settings

pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")

from app.gauge import model as gm  # noqa: E402  (PySide6 の有無を見てから読む)
from app.gauge.protocol import GaugeFrame, PartReading  # noqa: E402

HYBRID = "hybrid_measure"
USB = "realtime"

# 停止ファイルが置かれるまで待つだけの子。WorkerRunner.stop が停止ファイルを置くと抜ける。
_CHILD_WAITS_FOR_STOP_FILE = (
    "import os, pathlib, time\n"
    "stop = pathlib.Path(os.environ['APP_STOP_FILE'])\n"
    "while not stop.exists():\n"
    "    time.sleep(0.02)\n"
)


@pytest.fixture(scope="module")
def qt_app():
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def page(qt_app, settings):
    from app.shell.page_measure import MeasurePage

    page = MeasurePage(settings)
    yield page
    page.shutdown()


def _choose_input(page, role: str) -> None:
    """入力（USB カメラ 2 台／Mac＋Pixel）を選ぶ。"""
    page._input_mode.setCurrentIndex(1 if role == HYBRID else 0)
    assert page._runner.role == role


def _fake_start(monkeypatch, page, ok: bool = True) -> list[str]:
    """子プロセスを起動せずに、起動の成否と状態の変化だけを真似る。呼ばれた role を記録する。"""
    runner = page._runner
    calls: list[str] = []

    def start(settings, passthrough=None, module=None):
        calls.append(runner.role)
        runner.state_changed.emit("starting")
        if not ok:
            runner.state_changed.emit("stopped")
            return False
        runner.state_changed.emit("running")
        return True

    monkeypatch.setattr(runner, "start", start)
    return calls


def _frame(rep: int = 2, link: str = "connected") -> GaugeFrame:
    reading = PartReading(now=50.0, prev=40.0, band=(60.0, 80.0))
    return GaugeFrame(link=link, rep=rep, parts={"elbow_L": reading})


def _start_hybrid(monkeypatch, page) -> None:
    _choose_input(page, HYBRID)
    _fake_start(monkeypatch, page)
    page._main_button.click()


# ---------------------------------------------------------------------------
# 主ボタン
# ---------------------------------------------------------------------------


class TestMainButton:
    def test_main_button_swaps_label_with_state(self, page):
        assert page._main_button.text() == "計測を開始"
        for state, label in (("starting", "停止"), ("running", "停止"), ("stopped", "計測を開始")):
            page._runner.state_changed.emit(state)
            assert page._main_button.text() == label, state

    def test_main_button_stays_enabled_while_running(self, page):
        from app.core.qt import QtWidgets

        page._runner.state_changed.emit("running")
        assert page._main_button.isEnabled()
        buttons = [b.text() for b in page.findChildren(QtWidgets.QPushButton)]
        assert buttons.count("停止") == 1, f"主ボタンのほかに停止ボタンが残っている: {buttons}"

    def test_main_button_starts_then_stops(self, page, monkeypatch):
        from app import entry

        monkeypatch.setattr(
            entry, "worker_command",
            lambda role, passthrough=None, module=None: [sys.executable, "-c", _CHILD_WAITS_FOR_STOP_FILE],
        )
        _choose_input(page, USB)

        page._main_button.click()
        assert page.is_running, "主ボタンで開始していない"
        assert page._main_button.text() == "停止"

        page._main_button.click()  # 停止ファイルを置き、子が抜けるのを待つ
        assert not page.is_running, "主ボタンで停止していない"
        assert page._main_button.text() == "計測を開始"


# ---------------------------------------------------------------------------
# J の数値のスイッチ
# ---------------------------------------------------------------------------


class TestJoulesSwitch:
    def test_joules_switch_only_for_hybrid(self, page):
        _choose_input(page, USB)
        assert page._joules_switch.isHidden()
        _choose_input(page, HYBRID)
        assert not page._joules_switch.isHidden()
        _choose_input(page, USB)
        assert page._joules_switch.isHidden()

    def test_joules_switch_enabled_while_running(self, page, monkeypatch):
        _start_hybrid(monkeypatch, page)
        assert page._joules_switch.isEnabled()

    def test_switch_updates_setting_and_open_gauge(self, page, settings, monkeypatch):
        _start_hybrid(monkeypatch, page)
        assert page._joules_switch.isChecked() is True

        page._joules_switch.click()
        assert settings.get("GAUGE_SHOW_JOULES") is False
        assert page._gauge_window.gauge.state.show_joules is False

        page._joules_switch.click()
        assert settings.get("GAUGE_SHOW_JOULES") is True
        assert page._gauge_window.gauge.state.show_joules is True


# ---------------------------------------------------------------------------
# ゲージ窓の結線
# ---------------------------------------------------------------------------


class TestGaugeWindowWiring:
    def test_hybrid_start_opens_gauge_window_in_waiting_state(self, qt_app, monkeypatch):
        from app.shell.page_measure import MeasurePage

        settings = Settings()
        settings.set("GAUGE_SHOW_JOULES", False)
        page = MeasurePage(settings)
        try:
            assert page._joules_switch.isChecked() is False, "スイッチが設定の値で始まっていない"
            _start_hybrid(monkeypatch, page)
            window = page._gauge_window
            assert window.isVisible()
            assert window.gauge.state.phase is gm.Phase.WAITING
            assert window.gauge.state.frame is None
            assert window.gauge.state.show_joules is False, "開くときに設定の値を使っていない"
        finally:
            page.shutdown()

    def test_usb_start_does_not_open_gauge_window(self, page, monkeypatch):
        _choose_input(page, USB)
        calls = _fake_start(monkeypatch, page)
        page._main_button.click()
        assert calls == [USB]
        assert not page._gauge_window.isVisible()

    def test_failed_start_does_not_open_gauge_window(self, page, monkeypatch, tmp_path):
        from app import entry

        monkeypatch.setattr(
            entry, "worker_command",
            lambda role, passthrough=None, module=None: [str(tmp_path / "no_such_program")],
        )
        _choose_input(page, HYBRID)
        page._main_button.click()
        assert not page.is_running
        assert not page._gauge_window.isVisible()
        assert page._main_button.text() == "計測を開始"

    def test_gauge_frames_reach_the_window(self, page, monkeypatch):
        _start_hybrid(monkeypatch, page)
        frame = _frame(rep=3)
        page._runner.gauge_frame.emit(frame)
        state = page._gauge_window.gauge.state
        assert state.frame == frame
        assert state.phase is gm.Phase.RUNNING

    def test_finished_zero_shows_done_nonzero_shows_failed(self, page, monkeypatch):
        _start_hybrid(monkeypatch, page)
        page._runner.gauge_frame.emit(_frame())
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(0)
        assert page._gauge_window.gauge.state.phase is gm.Phase.DONE
        assert page._gauge_window.isVisible(), "終わっても窓は残す（被験者が最後の回を見られるように）"

        page._main_button.click()
        page._runner.gauge_frame.emit(_frame())
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(3)
        assert page._gauge_window.gauge.state.phase is gm.Phase.FAILED

    def test_shutdown_closes_gauge_window(self, page, monkeypatch):
        _start_hybrid(monkeypatch, page)
        assert page._gauge_window.isVisible()
        page.shutdown()
        assert not page._gauge_window.isVisible()

    def test_restart_resets_gauge_window(self, page, monkeypatch):
        _start_hybrid(monkeypatch, page)
        page._runner.gauge_frame.emit(_frame(rep=5))
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(0)
        page._gauge_window.close()  # 作業者が窓を閉じても、次の開始でまた開く

        page._main_button.click()
        state = page._gauge_window.gauge.state
        assert page._gauge_window.isVisible()
        assert state.phase is gm.Phase.WAITING
        assert state.frame is None
