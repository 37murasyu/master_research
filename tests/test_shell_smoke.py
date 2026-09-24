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


class TestAnalyzeArguments:
    """解析画面が子のスクリプトへ渡す引数。"""

    @pytest.fixture
    def page(self, qt_app, monkeypatch):
        from app.shell.page_analyze import AnalyzePage

        page = AnalyzePage(Settings())
        page.calls = []
        monkeypatch.setattr(page._runner, "start",
                            lambda settings, args, module=None: page.calls.append((args, module)) or True)
        yield page
        page.shutdown()

    def _choose(self, page, module: str):
        from app.shell.page_analyze import TASKS

        page._task_combo.setCurrentIndex(next(i for i, t in enumerate(TASKS) if t.module == module))

    def test_local_torque_passes_the_required_id(self, page, tmp_path):
        """「局所トルクの再計算」は必須の位置引数（計測の ID）を渡さず、必ず exit 2 で終わっていた。"""
        import compute_local_torque_offline

        self._choose(page, "compute_local_torque_offline")
        assert page._id_edit.isVisibleTo(page)
        page._input_edit.setText(str(tmp_path))
        page._id_edit.setText(" 0924_095256 ")
        page._run()

        [(args, module)] = page.calls
        assert (args, module) == (["0924_095256", "--base-dir", str(tmp_path)], "compute_local_torque_offline")
        # スクリプトの引数の解釈を通る（ファイルが無いので 1。引数の誤りなら SystemExit(2)）
        assert compute_local_torque_offline.main(args) == 1

    def test_local_torque_without_an_id_does_not_start(self, page, tmp_path):
        self._choose(page, "compute_local_torque_offline")
        page._input_edit.setText(str(tmp_path))
        page._run()
        assert page.calls == []
        assert "ID" in page._log.toPlainText()

    def test_the_id_field_is_only_for_tasks_that_need_it(self, page):
        self._choose(page, "stereo_triangulate_pose")
        assert not page._id_edit.isVisibleTo(page)

    def test_extra_options_keep_quoted_paths_with_spaces(self, page, tmp_path):
        self._choose(page, "stereo_triangulate_pose")
        page._input_edit.setText(str(tmp_path))
        page._extra_edit.setText('--out "/tmp/a b/out.csv" --fps 30')
        page._run()
        [(args, _module)] = page.calls
        assert args == ["--input-dir", str(tmp_path), "--out", "/tmp/a b/out.csv", "--fps", "30"]

    def test_an_unclosed_quote_is_reported_instead_of_starting(self, page, tmp_path):
        self._choose(page, "stereo_triangulate_pose")
        page._input_edit.setText(str(tmp_path))
        page._extra_edit.setText('--out "/tmp/a b')
        page._run()
        assert page.calls == []
        assert "引用符" in page._log.toPlainText()

    def test_windows_paths_keep_their_backslashes(self):
        from app.shell.page_analyze import split_options

        text = r'--out "C:\Users\a b\out.csv" --dir C:\data --fps 30'
        assert split_options(text, windows=True) == ["--out", r"C:\Users\a b\out.csv", "--dir", r"C:\data", "--fps", "30"]
        assert split_options("--out '/tmp/a b' --fps 30", windows=False) == ["--out", "/tmp/a b", "--fps", "30"]


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


class TestCameraProbeAtExit:
    def test_shutdown_leaves_no_running_probe_thread(self, qt_app, monkeypatch):
        """カメラ検出のスレッドが 3 秒を超えて動いていると、閉じた後に動いたまま破棄されて SIGABRT で落ちていた。

        閉じるときは検出の打ち切りを求め、スレッドが終わるまで待つ。
        """
        import threading
        import time

        from app.core import video_source
        from app.shell import page_calibrate
        from app.shell.page_calibrate import CalibratePage

        released = threading.Event()

        def slow_names():  # 遅いカメラの列挙（Windows の DSHOW の走査など）
            released.wait(10)
            return []

        opened = []
        monkeypatch.setattr(page_calibrate, "enumerate_camera_device_names", slow_names)
        monkeypatch.setattr(video_source, "open_source", lambda index: opened.append(index))
        timer = threading.Timer(3.5, released.set)  # 前の打ち切り（3 秒）より長く動かす

        page = CalibratePage(Settings())
        page._detect_button.click()
        assert page._probe.isRunning()
        timer.start()
        t0 = time.monotonic()
        try:
            page.shutdown()
            assert not page._probe.isRunning(), "閉じた後も検出のスレッドが動いている"
            assert time.monotonic() - t0 >= 3.0
            assert opened == [], "閉じるのに、カメラの走査を打ち切らずに続けた"
        finally:
            released.set()
            timer.cancel()
            page._probe.wait()


class TestOnePageAtATime:
    """どれかのページが実行中なら、ほかのページの開始は押せず、理由が出る。

    計測とキャリブレーションを同時に走らせられ、同じカメラや Pixel の接続を取り合っていた。
    """

    @pytest.fixture
    def window(self, qt_app, monkeypatch):
        from app.shell.main_window import MainWindow

        monkeypatch.setattr(MainWindow, "_save_settings", lambda self: None)
        window = MainWindow(Settings())
        yield window
        window.close()

    def test_a_running_measurement_blocks_the_other_pages(self, window):
        measure, calibrate, analyze = window._pages
        assert calibrate._start_button.isEnabled() and analyze._run_button.isEnabled()

        for state in ("starting", "running", "stopping"):
            measure._runner.state_changed.emit(state)
            assert not calibrate._start_button.isEnabled(), state
            assert not analyze._run_button.isEnabled(), state
            for page in (calibrate, analyze):
                assert page._blocked_label.isVisibleTo(page) and "計測" in page._blocked_label.text(), state
        assert not measure._blocked_label.isVisibleTo(measure), "実行中のページ自身には理由を出さない"

        measure._runner.state_changed.emit("stopped")
        assert calibrate._start_button.isEnabled() and analyze._run_button.isEnabled()
        assert not calibrate._blocked_label.isVisibleTo(calibrate)

    def test_a_running_calibration_blocks_the_measure_main_button_with_a_reason(self, window):
        measure, calibrate, _analyze = window._pages
        calibrate._runner.state_changed.emit("running")
        assert not measure._main_button.isEnabled()
        assert measure._start_blocked.isVisibleTo(measure) and "キャリブレーション" in measure._start_blocked.text()
        assert calibrate._stop_button.isEnabled(), "実行中のページは中止できる"

        calibrate._runner.state_changed.emit("stopped")
        assert measure._main_button.isEnabled()
        assert not measure._start_blocked.isVisibleTo(measure)

    def test_the_measurement_and_the_calibration_do_not_run_together(self, window, monkeypatch):
        import sys

        from app import entry

        started = []
        child = "import time\ntime.sleep(30)\n"
        monkeypatch.setattr(entry, "worker_command",
                            lambda role, passthrough=None, module=None: started.append(role) or [sys.executable, "-c", child])
        measure, calibrate, _analyze = window._pages
        try:
            measure._main_button.click()
            calibrate._start_button.click()
            assert started == ["realtime"]
            assert not calibrate.is_running
        finally:
            for page in window._pages:
                page._runner._process.kill()
                page._runner._process.waitForFinished(5000)


class TestStopping:
    """停止を求めてから子が終わるまで（state "stopping"）の画面。"""

    @pytest.mark.parametrize("page_name", ["calibrate", "analyze"])
    def test_start_and_abort_buttons_are_disabled_while_stopping(self, qt_app, page_name):
        from app.shell.page_analyze import AnalyzePage
        from app.shell.page_calibrate import CalibratePage

        page = {"calibrate": CalibratePage, "analyze": AnalyzePage}[page_name](Settings())
        try:
            start = page._start_button if page_name == "calibrate" else page._run_button
            page._runner.state_changed.emit("running")
            assert page._stop_button.isEnabled() and not start.isEnabled()
            page._runner.state_changed.emit("stopping")
            assert not page._stop_button.isEnabled(), "停止を待つ間に中止をもう一度押せる"
            assert not start.isEnabled(), "停止を待つ間に開始を押せる"
            assert page._badge.text() == "停止処理中"
            page._runner.state_changed.emit("stopped")
            assert start.isEnabled() and not page._stop_button.isEnabled()
        finally:
            page.shutdown()

    def test_closing_the_window_while_stopping_leaves_no_child(self, qt_app, monkeypatch):
        """停止を求めた後（子がまだ書き出し中）に窓を閉じても、子が終わるまで待って残さない。"""
        import sys

        from app import entry
        from app.core.qt import QtWidgets
        from app.shell.main_window import MainWindow

        child = ("import os, pathlib, time\n"
                 "stop = pathlib.Path(os.environ['APP_STOP_FILE'])\n"
                 "while not stop.exists():\n"
                 "    time.sleep(0.02)\n"
                 "time.sleep(0.5)\n")
        monkeypatch.setattr(entry, "worker_command",
                            lambda role, passthrough=None, module=None: [sys.executable, "-c", child])
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes)
        monkeypatch.setattr(MainWindow, "_save_settings", lambda self: None)

        window = MainWindow(Settings())
        measure = window._pages[0]
        assert measure._runner.start(Settings())
        measure._runner.stop()
        assert measure.is_running, "子が書き出し中のはず"

        window.close()
        assert not measure.is_running, "窓を閉じた後に子が残った"
