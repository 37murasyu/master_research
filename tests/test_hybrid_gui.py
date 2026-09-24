import pytest
from app import entry
from app.core.settings import Settings


def test_hybrid_workers_use_stop_files():
    for role in ("hybrid_calibrate", "hybrid_measure"):
        assert entry.resolve_module(role) == f"app.runners.{role}"
        assert entry.uses_stop_file(role)
    assert entry.uses_stop_file("realtime")
    assert not entry.uses_stop_file("calibrate")


def test_replay_is_its_own_worker_role():
    """記録の再生は独立の role。GUI は入力「記録の再生」でこれを起動し、停止ボタン（停止ファイル）で止める。"""
    assert entry.REPLAY_ROLE == "hybrid_replay"
    assert entry.REPLAY_ROLE in entry.ROLES
    assert entry.resolve_module(entry.REPLAY_ROLE) == "app.runners.hybrid_replay"
    assert entry.uses_stop_file(entry.REPLAY_ROLE)


def test_replay_arguments_reach_the_child_untouched():
    """何を流すか（計測フォルダ・範囲・速さ）は子の引数で渡す。アプリの入口の引数の解釈がそれを食べない。"""
    arguments = ["/somewhere/measure/20260923_000000_000000", "--from", "-3.0", "--speed", "0.0", "--to", "90"]
    command = entry.worker_command(entry.REPLAY_ROLE, arguments)
    parsed = entry.parse_args(command[command.index("--role"):])
    assert parsed.role == entry.REPLAY_ROLE
    assert parsed.passthrough == arguments


def _radio(page, role: str):
    """計測画面の、role の入力のラジオ（並びは MEASURE_INPUTS）。"""
    from app.shell.page_measure import MEASURE_INPUTS, measure_input

    return page._input_radios[MEASURE_INPUTS.index(measure_input(role))]


def test_input_switch_disables_during_run():
    pytest.importorskip("PySide6")
    from app.core.qt import QtWidgets
    from app.shell.page_measure import MeasurePage
    from app.shell.page_calibrate import CalibratePage

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    # 計測画面の入力はラジオ 3 つ（詳細設定の開示の中。R2-01）
    page = MeasurePage(Settings())
    radios = page._input_radios
    _radio(page, "hybrid_measure").click()
    assert page._runner.role == "hybrid_measure"
    _radio(page, entry.REPLAY_ROLE).click()
    assert page._runner.role == entry.REPLAY_ROLE
    page._on_state("running")
    assert not any(radio.isEnabled() for radio in radios)
    page._on_state("stopped")
    assert all(radio.isEnabled() for radio in radios)
    page.shutdown()

    page = CalibratePage(Settings())
    page._input_mode.setCurrentIndex(1)
    assert page._runner.role == "hybrid_calibrate"
    assert "hybrid" in page._output_label.text()
    page._on_state("running")
    assert not page._input_mode.isEnabled()
    page._on_state("stopped")
    assert page._input_mode.isEnabled()
    page.shutdown()
