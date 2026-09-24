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

    @pytest.mark.parametrize("content", [
        '{"values": {"SUBJECT_ID": "被験者3"}}'.encode("cp932"),
        b"[]",
        '{"values": {"BODY_MASS_KG": "65kg", "MP_THREADS": "auto", "HEADLESS": "TRUE", "SUBJECT_ID": 7}}'.encode(),
    ], ids=["cp932", "配列", "読めない値"])
    def test_window_opens_and_can_start_with_a_broken_settings_file(self, qt_app, tmp_path, monkeypatch, content):
        """手で書き換えた設定ファイルで、GUI が起動できない・開始で落ちる、にならないこと。"""
        from app import entry
        from app.core.qt import QtCore
        from app.shell import main_window as mw

        target = tmp_path / "settings.json"
        target.write_bytes(content)
        monkeypatch.setattr(mw.Settings, "default_path", classmethod(lambda cls: target))

        window = mw.MainWindow()
        try:
            assert window._settings.get("BODY_MASS_KG") == 65.0
            environment = QtCore.QProcessEnvironment()
            for key, value in entry.worker_environment(window._settings).items():
                environment.insert(key, value)  # 文字列でない値があると TypeError（開始で落ちていた）
        finally:
            window.close()


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


class TestEkfCalibrationTask:
    """S11 解析ページから、生 CSV を選んで EKF の較正プロファイルを作れる。

    新しいページは作らず、解析の種類に 1 項目足す（設計メモ 実装 5）。起動は
    ``--role script --module app.runners.tune_ekf`` の汎用経路で、生 CSV を位置引数で渡す。
    """

    def test_the_task_passes_the_raw_csv_to_the_tuning_command(self, qt_app, monkeypatch, tmp_path):
        from app.shell.page_analyze import TASKS, AnalyzePage

        task = next((t for t in TASKS if t.module == "app.runners.tune_ekf"), None)
        assert task is not None, "解析ページに EKF の較正の項目が無い"
        assert task.input_kind == "file" and task.input_option is None

        page = AnalyzePage(Settings())
        calls = []
        monkeypatch.setattr(page._runner, "start",
                            lambda settings, args, module=None: calls.append((args, module)) or True)
        page._task_combo.setCurrentIndex(TASKS.index(task))
        csv = str(tmp_path / "kpts3d_raw_0923_120000.csv")
        page._input_edit.setText(csv)
        page._run()
        page.shutdown()
        assert calls == [([csv], "app.runners.tune_ekf")]

    def test_the_module_resolves_as_a_script(self):
        from app import entry

        assert entry.resolve_module("script", "app.runners.tune_ekf") == "app.runners.tune_ekf"


class TestStopDirectory:
    """停止ファイル（§3-2）を置く一時ディレクトリは、計測のときだけ作り、残さない。"""

    def test_only_the_measurement_gets_a_stop_directory(self, qt_app, monkeypatch):
        import sys
        import tempfile

        from app import entry
        from app.runners.worker import WorkerRunner

        made = []
        real_mkdtemp = tempfile.mkdtemp
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix="": made.append(prefix) or real_mkdtemp(prefix=prefix))
        monkeypatch.setattr(entry, "worker_command", lambda role, passthrough=None, module=None: [sys.executable, "-c", "pass"])
        runner = WorkerRunner("script")
        assert runner.start(Settings(), module="dummy")
        runner._process.waitForFinished(5000)
        assert made == [], "停止ファイルを見ない役割にも停止用のディレクトリを作っている"

    def test_a_failed_start_leaves_no_directory(self, qt_app, monkeypatch, tmp_path):
        import tempfile

        from app import entry
        from app.runners.worker import WorkerRunner

        real_mkdtemp = tempfile.mkdtemp
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix="": real_mkdtemp(prefix=prefix, dir=tmp_path))
        monkeypatch.setattr(entry, "worker_command",
                            lambda role, passthrough=None, module=None: [str(tmp_path / "no_such_program")])
        runner = WorkerRunner("realtime")
        assert runner.start(Settings()) is False
        assert not list(tmp_path.glob("wt_stop_*")), "起動に失敗したのに停止用のディレクトリが残った"
