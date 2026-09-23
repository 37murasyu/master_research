"""計測画面（実験者用）で使う部品（``app.shell.controls``）の試験。

対象は 4 つ: ``ToggleSwitch``（溝・つまみ・文字を自前で描くスイッチ）、
``Disclosure``（開閉できる見出しと中身）、``CountBadge``（0 のとき隠れる件数）、
``StatusText``（記号と文字の組で状態を示すラベル）。

見出しは白地（Qt の標準パレット）の上に置く前提で、色は
``app/shell/theme.py`` から取る。constraints.md の「色だけで状態を示さない」
を守っているか（記号や位置など、色以外の手がかりが必ず添うか）も併せて確かめる。
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")


@pytest.fixture(scope="module")
def qt_app():
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class TestToggleSwitch:
    def test_switch_toggles_on_click_and_space(self, qt_app):
        from pyqtgraph.Qt import QtCore, QtTest

        from app.shell.controls import ToggleSwitch

        switch = ToggleSwitch("自動更新")
        assert switch.isChecked() is False

        QtTest.QTest.mouseClick(switch, QtCore.Qt.LeftButton)
        assert switch.isChecked() is True

        switch.setFocus()
        QtTest.QTest.keyClick(switch, QtCore.Qt.Key_Space)
        assert switch.isChecked() is False

    def test_switch_has_text_and_accessible_name(self, qt_app):
        from app.core.qt import QtCore

        from app.shell.controls import ToggleSwitch

        switch = ToggleSwitch("自動更新")
        assert switch.text() == "自動更新"
        assert switch.accessibleName() != ""
        # 位置(つまみの左右)という、色以外の手がかりが常に付く部品であること
        # （溝とつまみを自前で描くための土台がある）を確かめる。
        assert switch.isCheckable() is True
        assert switch.focusPolicy() == QtCore.Qt.StrongFocus


class TestDisclosure:
    def test_disclosure_shows_content_when_opened(self, qt_app):
        from app.core.qt import QtWidgets

        from app.shell.controls import Disclosure

        content = QtWidgets.QLabel("中身")
        disclosure = Disclosure("詳細", content)

        # 実際にウィンドウとして出さない試験なので isVisible() ではなく、
        # 「disclosure が出れば中身も出るか」を isVisibleTo で確かめる。
        assert disclosure.is_open() is False
        assert content.isVisibleTo(disclosure) is False

        disclosure.set_open(True)
        assert disclosure.is_open() is True
        assert content.isVisibleTo(disclosure) is True

        disclosure.set_open(False)
        assert disclosure.is_open() is False
        assert content.isVisibleTo(disclosure) is False

    def test_disclosure_badge_shows_count(self, qt_app):
        from app.core.qt import QtWidgets

        from app.shell.controls import Disclosure

        content = QtWidgets.QLabel("中身")
        disclosure = Disclosure("詳細", content)
        disclosure.set_badge(3)
        assert "3" in disclosure._badge.text()


class TestCountBadge:
    def test_badge_shows_count_and_hides_at_zero(self, qt_app):
        from app.shell.controls import CountBadge

        badge = CountBadge()
        assert badge.isVisible() is False

        badge.set_count(5)
        assert badge.isVisible() is True
        assert "5" in badge.text()
        assert "✕" in badge.text()

        badge.set_count(0)
        assert badge.isVisible() is False


class TestStatusText:
    def test_status_text_pairs_symbol_and_words(self, qt_app):
        from app.shell.controls import StatusText

        status = StatusText()

        status.set_status("running", rep=7)
        text = status.text()
        assert "●" in text
        assert "計測中" in text
        assert "7" in text

        status.set_status("stopped")
        assert "停止中" in status.text()

        status.set_status("success")
        assert "✓" in status.text()
        assert "正常終了" in status.text()

        status.set_status("error")
        assert "✕" in status.text()
        assert "異常終了" in status.text()

        status.set_link(True)
        assert "●" in status.text()
        assert "Pixel 接続" in status.text()

        status.set_link(False)
        assert "○" in status.text()
        assert "Pixel 接続待ち" in status.text()
