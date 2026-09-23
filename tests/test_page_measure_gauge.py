"""計測画面が、混成（Mac＋Pixel）の計測でゲージ窓を開き、子の ``@@GAUGE`` の行を渡すことを確かめる。

**なぜこのテストがあるか。**

子プロセスは押し上げの仕事と論文の閾値を ``@@GAUGE`` の行で出し、``WorkerRunner.gauge_frame`` がそれを GaugeFrame に
して流す。計測画面がゲージ窓を開いてつながないと、本番（GUI）では被験者に何も見えない（2026-09-24 朝の最小の配線。
設計書 §5.2 の作り込みは UI 担当が別に入れる）。USB カメラ 2 台の計測ではゲージ窓を開かない（旧ゲージは子が自分で描く）。
"""

from __future__ import annotations

import pytest

from app.core.settings import Settings
from app.gauge.protocol import GaugeFrame, PartReading
from app.shell import page_measure


class FakeWindow:
    instances: list["FakeWindow"] = []

    def __init__(self, *, show_joules=True, parent=None):
        self.calls = [("init", show_joules)]
        FakeWindow.instances.append(self)

    def begin(self, *, show_joules, avoid_screen=None):
        self.calls.append(("begin", show_joules))

    def set_frame(self, frame):
        self.calls.append(("frame", frame.rep))

    def finish(self, exit_code):
        self.calls.append(("finish", exit_code))


@pytest.fixture(scope="module")
def qapp():
    from app.core.qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def page(qapp, monkeypatch):
    FakeWindow.instances.clear()
    monkeypatch.setattr(page_measure, "GaugeWindow", FakeWindow)
    return page_measure.MeasurePage(Settings())


def _frame(rep):
    return GaugeFrame(link="connected", rep=rep, source="measure",
                      parts={"elbow_L": PartReading(now=10.0, prev=None, band=(47.0, 57.0), w1rm=66.0)})


def test_hybrid_measurement_opens_the_gauge_and_forwards_frames(page):
    page._input_mode.setCurrentIndex(1)
    page._open_gauge()
    page._runner.gauge_frame.emit(_frame(2))
    page._runner.finished.emit(0)
    window = FakeWindow.instances[-1]
    assert ("frame", 2) in window.calls
    assert window.calls[-1] == ("finish", 0)
    assert any(call[0] == "begin" for call in window.calls)


def test_usb_measurement_does_not_open_the_gauge(page):
    page._input_mode.setCurrentIndex(0)
    page._open_gauge()
    page._runner.gauge_frame.emit(_frame(1))
    assert FakeWindow.instances == []
