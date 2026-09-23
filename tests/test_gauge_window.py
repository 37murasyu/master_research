"""被験者ゲージの窓（``app.gauge.window``）の試験。

被験者が見るだけの画面なので見た目は ``test_gauge_widget.py`` に任せ、ここでは
窓としての振る舞い（第 2 モニタの選び方・閉じても隠れるだけ・アプリの終了を
妨げない・部品を置かない・``begin`` が J の表示を反映すること）だけを確かめる。

``choose_screen`` は Qt に依存しない（identity 比較だけの）純粋関数なので、
偽の画面オブジェクト（``object()``）で試験する。offscreen 環境の画面は
常に 1 つなので、全画面の経路（2 台目のモニタがある場合）は ``choose_screen``
の選び方だけをここで確かめ、``GaugeWindow.present`` 自体の全画面分岐は
実機（Gauge_display.py と同じ環境）まで試験しない。
"""

from __future__ import annotations

import pytest

from app.gauge import model as gm
from app.gauge.protocol import GaugeFrame, PartReading
from app.gauge.window import GaugeWindow, choose_screen

pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")


@pytest.fixture(scope="module")
def qt_app():
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class TestChooseScreen:
    def test_choose_screen_avoids_the_operator_screen(self):
        operator = object()
        second = object()
        third = object()
        assert choose_screen([operator, second, third], operator) is second

    def test_choose_screen_returns_none_with_one_screen(self):
        only = object()
        assert choose_screen([only], None) is None


class TestGaugeWindow:
    def test_single_screen_opens_1280x720_window(self, qt_app):
        window = GaugeWindow()
        window.begin(show_joules=True)
        try:
            assert (window.width(), window.height()) == (1280, 720)
            assert not window.isFullScreen()
        finally:
            window.close()

    def test_window_does_not_keep_the_app_alive(self, qt_app):
        from app.core.qt import QtCore

        window = GaugeWindow()
        assert window.testAttribute(QtCore.Qt.WA_QuitOnClose) is False
        window.close()

    def test_begin_resets_to_waiting_and_applies_joules(self, qt_app):
        window = GaugeWindow()
        try:
            frame = GaugeFrame(link="connected", rep=1, source="measure", parts={"elbow_L": PartReading(now=10.0)})
            window.set_frame(frame)
            window.set_show_joules(True)

            window.begin(show_joules=False)

            assert window.gauge.state.phase is gm.Phase.WAITING
            assert window.gauge.state.frame is None
            assert window.gauge.state.show_joules is False
        finally:
            window.close()

    def test_has_no_controls(self, qt_app):
        from app.core.qt import QtWidgets
        from app.gauge.widget import GaugeWidget

        window = GaugeWindow()
        try:
            assert window.findChildren(QtWidgets.QAbstractButton) == []
            assert isinstance(window.gauge, GaugeWidget)
        finally:
            window.close()

    def test_closed_window_can_begin_again(self, qt_app):
        window = GaugeWindow()
        window.begin(show_joules=True)
        window.close()
        assert not window.isVisible()

        window.begin(show_joules=True)
        try:
            assert window.isVisible()
        finally:
            window.close()

    def test_public_api_names(self, qt_app):
        window = GaugeWindow()
        try:
            for name in ("set_frame", "set_show_joules", "begin", "finish"):
                assert callable(getattr(window, name))
        finally:
            window.close()
