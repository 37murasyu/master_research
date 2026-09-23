import pytest
from app import entry
from app.core.settings import Settings


def test_hybrid_workers_use_stop_files():
    for role in ("hybrid_calibrate", "hybrid_measure"):
        assert entry.resolve_module(role) == f"app.runners.{role}"
        assert entry.uses_stop_file(role)
    assert entry.uses_stop_file("realtime")
    assert not entry.uses_stop_file("calibrate")


def test_input_switch_disables_during_run():
    pytest.importorskip("PySide6")
    from app.core.qt import QtWidgets
    from app.shell.page_measure import MeasurePage
    from app.shell.page_calibrate import CalibratePage

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    # 計測画面の入力はラジオ 2 つ（詳細設定の開示の中。R2-01）
    page = MeasurePage(Settings())
    page._input_hybrid.click()
    assert page._runner.role == "hybrid_measure"
    page._on_state("running")
    assert not page._input_usb.isEnabled() and not page._input_hybrid.isEnabled()
    page._on_state("stopped")
    assert page._input_usb.isEnabled() and page._input_hybrid.isEnabled()
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
