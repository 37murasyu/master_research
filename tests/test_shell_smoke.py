"""GUI シェルが組み立てられることを確かめる煙テスト。

見た目は検証しない。狙いは「import できる」「ウィジェットの生成で落ちない」
「子プロセスを残さず閉じられる」の 3 点。GUI は壊れても静かなことが多く、
起動して初めて分かる類の失敗（シグナル名の綴り間違い、Qt の API 差など）を
早く捕まえる。

QT_QPA_PLATFORM=offscreen で画面を出さずに動かす。conftest.py で設定している。
"""

from __future__ import annotations

import pytest

from app.core.settings import Settings

pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")


@pytest.fixture(scope="module")
def qt_app():
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class TestPagesBuild:
    def test_measure_page_builds(self, qt_app):
        from app.shell.page_measure import MeasurePage

        page = MeasurePage(Settings())
        assert page.is_running is False
        page.shutdown()

    def test_calibrate_page_builds(self, qt_app):
        from app.shell.page_calibrate import CalibratePage

        page = CalibratePage(Settings())
        assert page.is_running is False
        page.shutdown()

    def test_analyze_page_builds(self, qt_app):
        from app.shell.page_analyze import AnalyzePage

        page = AnalyzePage(Settings())
        assert page.is_running is False
        page.shutdown()


class TestMainWindow:
    def test_window_builds_with_three_pages(self, qt_app):
        from app.shell.main_window import MainWindow

        window = MainWindow(Settings())
        assert window._stack.count() == 3
        assert window._nav.count() == 3
        window.close()

    def test_navigation_switches_pages(self, qt_app):
        from app.shell.main_window import MainWindow

        window = MainWindow(Settings())
        for row in range(window._nav.count()):
            window._nav.setCurrentRow(row)
            assert window._stack.currentIndex() == row
        window.close()

    def test_closing_saves_settings(self, qt_app, tmp_path, monkeypatch):
        from app.shell import main_window as mw

        target = tmp_path / "settings.json"
        monkeypatch.setattr(mw.Settings, "default_path", classmethod(lambda cls: target))

        window = mw.MainWindow()
        window._settings.set("DEMO_MONO_GAUGE_ON", True)
        window.close()

        assert target.is_file(), "終了時に設定が保存されていない"
        assert Settings.load(target).get("DEMO_MONO_GAUGE_ON") is True


class TestSettingsForm:
    def test_shows_only_ui_visible_settings(self, qt_app):
        from app.core.qt import QtWidgets
        from app.core.settings import SCHEMA
        from app.shell.widgets import SettingsForm

        form = SettingsForm(Settings())
        expected = sum(1 for s in SCHEMA.values() if s.ui_visible)
        rows = sum(
            box.layout().rowCount() for box in form.findChildren(QtWidgets.QGroupBox)
        )
        assert rows == expected

    def test_editing_updates_the_settings_object(self, qt_app):
        from app.shell.widgets import SettingsForm

        settings = Settings()
        form = SettingsForm(settings)
        form._on_change("DEMO_MONO_GAUGE_ON", True)
        assert settings.get("DEMO_MONO_GAUGE_ON") is True


class TestLogView:
    def test_line_count_is_capped(self, qt_app):
        """計測は毎フレーム print する。上限が無いと長時間でメモリを食い潰す。"""
        from app.shell.widgets import LogView

        view = LogView()
        for i in range(view.MAX_BLOCKS + 500):
            view.append_text(f"行 {i}\n")
        assert view.blockCount() <= view.MAX_BLOCKS + 1
