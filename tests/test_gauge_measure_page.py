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
    """入力（USB カメラ 2 台／Mac＋Pixel）のラジオを押す。"""
    (page._input_hybrid if role == HYBRID else page._input_usb).click()
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


# ---------------------------------------------------------------------------
# 見出しの状態（点と文字の組。色だけにしない）
# ---------------------------------------------------------------------------


class TestHeaderStatus:
    def test_header_shows_amber_dot_and_rep_while_running(self, page, monkeypatch):
        from app.shell import theme

        assert page._run_status.text() == "停止中"

        _start_hybrid(monkeypatch, page)
        page._runner.gauge_frame.emit(_frame(rep=6))
        text = page._run_status.text()
        assert "●" in text and theme.AMBER in text, "琥珀の点が無い"
        assert "計測中 7 回目" in text, "回数は「完了した回数＋1」"

    def test_header_shows_no_rep_for_usb(self, page, monkeypatch):
        _choose_input(page, USB)
        _fake_start(monkeypatch, page)
        page._main_button.click()
        text = page._run_status.text()
        assert "計測中" in text
        assert "回目" not in text

    def test_header_shows_link_only_for_hybrid(self, page, monkeypatch):
        _choose_input(page, USB)
        _fake_start(monkeypatch, page)
        page._main_button.click()
        assert page._link_status.isHidden(), "USB の計測で Pixel の接続を出している"
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(0)

        _start_hybrid(monkeypatch, page)
        assert not page._link_status.isHidden()
        assert page._link_status.text() == "○ Pixel 接続待ち"
        assert "回目" not in page._run_status.text(), "つながる前に回数を出している"

        page._runner.gauge_frame.emit(_frame(rep=0, link="connected"))
        assert "●" in page._link_status.text() and "Pixel 接続" in page._link_status.text()
        assert "接続待ち" not in page._link_status.text()
        assert "計測中 1 回目" in page._run_status.text()

        page._runner.gauge_frame.emit(_frame(rep=1, link="waiting"))
        assert page._link_status.text() == "○ Pixel 接続待ち", "切れたら接続待ちに戻す"

        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(0)
        assert page._link_status.isHidden(), "終わった後も接続を出している"

    def test_header_after_finish_shows_result(self, page, monkeypatch):
        _start_hybrid(monkeypatch, page)
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(0)
        assert page._run_status.text() == "✓ 正常終了"

        page._main_button.click()
        assert "計測中" in page._run_status.text()
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(1)
        text = page._run_status.text()
        assert "✕" in text and "異常終了" in text

    def test_header_shows_stopped_after_failed_start(self, page, monkeypatch):
        _choose_input(page, HYBRID)
        _fake_start(monkeypatch, page, ok=False)
        page._main_button.click()
        assert page._run_status.text() == "停止中"
        assert page._link_status.isHidden()

    def test_status_badge_is_hidden_on_measure_page(self, page):
        assert page._badge.isHidden()
        for state in ("starting", "running", "stopped"):
            page._runner.state_changed.emit(state)
            assert page._badge.isHidden(), state

    def test_other_pages_keep_their_badge(self, qt_app):
        from app.shell.page_calibrate import CalibratePage

        page = CalibratePage(Settings())
        try:
            assert not page._badge.isHidden()
        finally:
            page.shutdown()


# ---------------------------------------------------------------------------
# 実験者用の詳細設定（開示）と、説明文の削除
# ---------------------------------------------------------------------------


def _row_names(form) -> set[str]:
    from app.core.qt import QtWidgets

    return {label.text() for label in form.findChildren(QtWidgets.QLabel)}


def _folder_link_url(page, monkeypatch):
    """「出力フォルダ」リンクを押し、開こうとした URL を返す（実際には開かない）。"""
    from app.core.qt import QtGui

    opened = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    page._output_link.linkActivated.emit("#")
    assert len(opened) == 1
    return opened[0]


class TestAdvancedSettings:
    def test_input_is_a_pair_of_radio_buttons(self, page):
        from app.core.qt import QtWidgets

        radios = page.findChildren(QtWidgets.QRadioButton)
        assert [r.text() for r in radios] == ["USB カメラ 2 台", "Mac＋Pixel"]
        group = radios[0].group()
        assert group is not None and group is radios[1].group() and group.exclusive()
        assert page._input_usb.isChecked() and page._runner.role == USB
        assert not page.findChildren(QtWidgets.QComboBox), "入力のコンボボックスが残っている（n=2 は R2-03）"

    def test_radio_switches_role(self, page):
        page._input_hybrid.click()
        assert page._runner.role == HYBRID
        assert not page._joules_switch.isHidden()
        page._input_usb.click()
        assert page._runner.role == USB
        assert page._joules_switch.isHidden()

    def test_advanced_settings_are_in_a_closed_disclosure(self, page):
        from app.shell.controls import Disclosure

        advanced = page._advanced
        assert isinstance(advanced, Disclosure)
        assert advanced._button.text() == "実験者用の詳細設定"
        assert not advanced.is_open(), "既定で閉じていない"
        for widget in (page._input_usb, page._input_hybrid, page._subject_edit, page._body_mass):
            assert advanced.isAncestorOf(widget)

        dev = page._dev
        assert isinstance(dev, Disclosure) and advanced.isAncestorOf(dev)
        assert dev._button.text() == "開発・診断用"
        assert not dev.is_open()
        assert dev.isAncestorOf(page._form)

        advanced.set_open(True)
        assert page._subject_edit.isVisibleTo(page)

    def test_subject_and_body_mass_rows_edit_settings(self, qt_app):
        from app.core.qt import QtWidgets
        from app.shell.page_measure import MeasurePage

        settings = Settings({"SUBJECT_ID": "3", "BODY_MASS_KG": "70"})
        page = MeasurePage(settings)
        try:
            assert page._subject_edit.text() == "3"
            mass = page._body_mass
            assert isinstance(mass, QtWidgets.QDoubleSpinBox)
            assert mass.value() == pytest.approx(70.0)
            assert (mass.decimals(), mass.minimum(), mass.maximum()) == (1, 20.0, 200.0)

            # 欄の直後に単位（R9-08）
            row = mass.parentWidget().layout()
            unit = row.itemAt(row.indexOf(mass) + 1).widget()
            assert isinstance(unit, QtWidgets.QLabel) and unit.text() == "kg"

            page._subject_edit.setText("7")
            assert settings.get("SUBJECT_ID") == "7"
            mass.setValue(72.5)
            assert settings.get("BODY_MASS_KG") == pytest.approx(72.5)
        finally:
            page.shutdown()

    def test_nested_form_excludes_dedicated_rows(self, page):
        from app.core.settings import SCHEMA
        from app.shell.widgets import SettingsForm

        names = _row_names(page._form)
        assert "SUBJECT_ID" not in names and "BODY_MASS_KG" not in names
        assert "DEMO_MONO_GAUGE_ON" in names

        visible = {s.name for s in SCHEMA.values() if s.ui_visible}
        form = SettingsForm(Settings(), exclude=frozenset({"SUBJECT_ID"}))
        assert _row_names(form) & visible == visible - {"SUBJECT_ID"}

    def test_settings_disabled_while_running_with_reason(self, page):
        editors = (page._input_usb, page._input_hybrid, page._subject_edit, page._body_mass, page._form)
        assert page._locked_reason.isHidden()

        page._runner.state_changed.emit("running")
        for widget in editors:
            assert not widget.isEnabled(), widget
        assert not page._locked_reason.isHidden()
        assert page._locked_reason.text() == "計測中は変更できません"
        assert page._advanced.isAncestorOf(page._locked_reason), "理由が詳細設定の中に無い"
        assert page._advanced._button.isEnabled(), "計測中も中身は見られる"

        page._runner.state_changed.emit("stopped")
        for widget in editors:
            assert widget.isEnabled(), widget
        assert page._locked_reason.isHidden()

    def test_no_explanatory_text_left(self, page):
        from app.core.qt import QtWidgets

        page._advanced.set_open(True)
        page._dev.set_open(True)
        texts = [label.text() for label in page.findChildren(QtWidgets.QLabel)]
        for text in texts:
            for phrase in ("デモ用", "出力先", "既定から変更", "カーソル"):
                assert phrase not in text, f"説明文が残っている: {text!r}"
            assert "。" not in text, f"文になっている（説明文）: {text!r}"
        assert not hasattr(page, "_output_label")

    def test_output_folder_link_after_finish(self, page, monkeypatch):
        from app.core.qt import QtCore
        from app.core.settings import measurement_output_dir
        from app.hybrid import paths as hybrid_paths

        assert page._output_link.isHidden(), "実行前にリンクを出している"

        _choose_input(page, USB)
        _fake_start(monkeypatch, page)
        page._main_button.click()
        assert page._output_link.isHidden()
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(0)
        assert not page._output_link.isHidden()
        assert "出力フォルダ" in page._output_link.text()
        url = _folder_link_url(page, monkeypatch)
        assert url == QtCore.QUrl.fromLocalFile(str(measurement_output_dir()))

        _start_hybrid(monkeypatch, page)
        assert page._output_link.isHidden(), "次の実行が始まってもリンクが残っている"
        page._runner.state_changed.emit("stopped")
        page._runner.finished.emit(1)
        assert not page._output_link.isHidden(), "異常終了でも途中までの出力はある"
        url = _folder_link_url(page, monkeypatch)
        assert url == QtCore.QUrl.fromLocalFile(str(hybrid_paths.measurement_root()))


# ---------------------------------------------------------------------------
# 使う校正の日時と「変更」リンク
# ---------------------------------------------------------------------------


def _page_with_calibration_root(monkeypatch, root):
    from app.hybrid import paths as hybrid_paths
    from app.shell.page_measure import MeasurePage

    monkeypatch.setattr(hybrid_paths, "calibration_root", lambda: root)
    return MeasurePage(Settings())


def _write_latest(root, directory: str) -> None:
    import json

    (root / "latest.json").write_text(json.dumps({"directory": directory}), encoding="utf-8")


class TestCalibrationTime:
    def test_calibration_time_is_read_from_latest_json(self, qt_app, monkeypatch, tmp_path):
        _write_latest(tmp_path, "20260923_215130_123456")
        page = _page_with_calibration_root(monkeypatch, tmp_path)
        try:
            assert page._calibration_time.text() == "2026-09-23 21:51"
        finally:
            page.shutdown()

    @pytest.mark.parametrize("content", [None, "{", '{"directory": "latest"}', '{"other": 1}'])
    def test_without_a_readable_calibration_it_says_uncalibrated(self, qt_app, monkeypatch, tmp_path, content):
        if content is not None:
            (tmp_path / "latest.json").write_text(content, encoding="utf-8")
        page = _page_with_calibration_root(monkeypatch, tmp_path)
        try:
            assert page._calibration_time.text() == "未校正"
        finally:
            page.shutdown()

    def test_calibration_time_is_reread_when_the_page_is_shown(self, qt_app, monkeypatch, tmp_path):
        page = _page_with_calibration_root(monkeypatch, tmp_path)
        try:
            page.show()
            assert page._calibration_time.text() == "未校正"
            page.hide()
            _write_latest(tmp_path, "20260924_080512_000001")  # キャリブレーション画面で校正した
            page.show()
            assert page._calibration_time.text() == "2026-09-24 08:05"
        finally:
            page.shutdown()
            page.close()

    def test_calibration_row_only_for_hybrid(self, page):
        page._advanced.set_open(True)
        _choose_input(page, USB)
        assert not page._calibration_time.isVisibleTo(page), "USB の計測は混成の校正を使わない"
        assert not page._calibration_link.isVisibleTo(page)
        _choose_input(page, HYBRID)
        assert page._calibration_time.isVisibleTo(page)
        assert page._calibration_link.isVisibleTo(page)
        assert page._advanced.isAncestorOf(page._calibration_link)

    def test_page_reads_latest_json_without_calibration_io(self):
        from pathlib import Path

        import app.shell.page_measure as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "calibration_io" not in source, "GUI が校正の読み込み（numpy・cv2 側）を import している"

    def test_change_link_requests_calibration_page(self, qt_app, monkeypatch, tmp_path):
        from app.shell import main_window as mw

        monkeypatch.setattr(mw.Settings, "default_path", classmethod(lambda cls: tmp_path / "settings.json"))
        window = mw.MainWindow(Settings())
        try:
            page = window._pages[0]
            requested = []
            page.calibration_requested.connect(lambda: requested.append(True))
            page._calibration_link.linkActivated.emit("#")
            assert requested == [True]
            assert window._nav.currentRow() == 1
            assert window._stack.currentIndex() == 1
        finally:
            window.close()


# ---------------------------------------------------------------------------
# 計算を壊す設定の件数バッジ・ログの「未実行」・スイッチの即時保存
# ---------------------------------------------------------------------------


class TestBrokenFlagBadge:
    def test_badge_counts_enabled_broken_flags(self, qt_app):
        from app.shell.page_measure import MeasurePage

        settings = Settings()
        settings.set("DEMO_MONO_GAUGE_ON", True)
        page = MeasurePage(settings)
        try:
            assert page._advanced._badge.text() == "✕ 1"
            assert not page._advanced._badge.isHidden()
            settings.set("E_LPF_NATIVE_ON", True)
            page._form.changed.emit()
            assert page._advanced._badge.text() == "✕ 2"
            assert page._dev._badge.text() == "✕ 2", "中の開示にも、どこを見ればよいかを出す"
        finally:
            page.shutdown()

    def test_badge_is_hidden_when_every_flag_is_at_its_default(self, page):
        assert page._advanced._badge.isHidden()
        assert page._dev._badge.isHidden()

    def test_badge_follows_the_nested_form(self, page):
        from app.core.qt import QtWidgets

        # 行の見出しは設定名。DEMO_MONO_CAM0_ONLY のチェックボックスを押す
        for form in page._form.findChildren(QtWidgets.QFormLayout):
            for row in range(form.rowCount()):
                label = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
                if label is not None and label.widget().text() == "DEMO_MONO_CAM0_ONLY":
                    form.itemAt(row, QtWidgets.QFormLayout.FieldRole).widget().click()
        assert page._settings.get("DEMO_MONO_CAM0_ONLY") is True
        assert page._advanced._badge.text() == "✕ 1"


class TestLogPlaceholder:
    def test_log_shows_placeholder_before_first_run(self, page):
        assert page._log.placeholderText() == "未実行"
        assert page._log.toPlainText() == ""


class TestImmediateSave:
    def test_switch_emits_settings_edited(self, page):
        edited = []
        page.settings_edited.connect(lambda: edited.append(True))
        page._joules_switch.click()
        assert edited == [True]

    def test_switch_saves_immediately(self, qt_app, monkeypatch, tmp_path):
        import json

        from app.shell import main_window as mw

        path = tmp_path / "settings.json"
        monkeypatch.setattr(mw.Settings, "default_path", classmethod(lambda cls: path))
        window = mw.MainWindow(Settings())
        try:
            page = window._pages[0]
            before = bool(window._settings.get("GAUGE_SHOW_JOULES"))
            page._joules_switch.click()
            assert path.is_file(), "閉じる前に保存されていない"
            saved = json.loads(path.read_text(encoding="utf-8"))["values"]
            assert Settings(saved).get("GAUGE_SHOW_JOULES") is (not before)
        finally:
            window.close()
