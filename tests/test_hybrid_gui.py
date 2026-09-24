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
    assert "hybrid_replay" in entry.ROLES
    assert entry.resolve_module("hybrid_replay") == "app.runners.hybrid_replay"
    assert entry.uses_stop_file("hybrid_replay")
    assert entry.worker_command("hybrid_replay")[-2:] == ["--role", "hybrid_replay"]
    from app.hybrid.replay import REPLAY_ENV

    assert REPLAY_ENV.startswith(entry.REPLAY_ENV_PREFIX), "entry が落とす名前と再生が読む名前が食い違っている"


@pytest.mark.parametrize("role", ["hybrid_measure", "realtime", "hybrid_calibrate"])
def test_parent_replay_env_does_not_reach_other_workers(monkeypatch, role):
    """親のシェルに残った ``export HYBRID_REPLAY=...`` が、再生でない子へ届かない（本番の計測が黙って再生にならない）。"""
    monkeypatch.setenv("HYBRID_REPLAY", "/somewhere/measure/20260923_000000_000000")
    monkeypatch.setenv("HYBRID_REPLAY_FROM", "20")
    monkeypatch.setenv("HYBRID_REPLAY_SOMETHING", "1")
    settings = Settings()
    settings.set("HYBRID_REPLAY", "/chosen/in/the/gui")
    env = entry.worker_environment(settings, role=role)
    assert not [name for name in env if name.startswith("HYBRID_REPLAY")]


def test_replay_worker_reads_the_folder_from_settings_not_the_shell(monkeypatch):
    """再生の子へは設定（画面で選んだフォルダ）を渡す。親のシェルの値は使わない。"""
    monkeypatch.setenv("HYBRID_REPLAY", "/from/the/shell")
    monkeypatch.setenv("HYBRID_REPLAY_FROM", "20")
    monkeypatch.setenv("HYBRID_REPLAY_SOMETHING", "1")
    settings = Settings()
    settings.set("HYBRID_REPLAY", "/chosen/in/the/gui")
    env = entry.worker_environment(settings, role="hybrid_replay")
    assert env["HYBRID_REPLAY"] == "/chosen/in/the/gui"
    assert env["HYBRID_REPLAY_FROM"] == settings.as_env()["HYBRID_REPLAY_FROM"]
    assert "HYBRID_REPLAY_SOMETHING" not in env


def test_input_switch_disables_during_run():
    pytest.importorskip("PySide6")
    from app.core.qt import QtWidgets
    from app.shell.page_measure import MeasurePage
    from app.shell.page_calibrate import CalibratePage

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    # 計測画面の入力はラジオ 3 つ（詳細設定の開示の中。R2-01）
    page = MeasurePage(Settings())
    radios = (page._input_usb, page._input_hybrid, page._input_replay)
    page._input_hybrid.click()
    assert page._runner.role == "hybrid_measure"
    page._input_replay.click()
    assert page._runner.role == "hybrid_replay"
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
