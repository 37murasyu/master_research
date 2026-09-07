"""エントリポイントと計測ワーカーの起動方法を検証する。

PyInstaller で凍結すると ``python master_research_code.py`` は実行できない
（配布物に Python が無い）。そこでアプリは ``sys.executable`` に
``--role=realtime`` を付けて**自分自身を再起動**し、その役割で
既存の計測スクリプトを走らせる。

開発時は ``python -m app`` が ``python -m app --role realtime`` を起動するので、
**開発と配布で同じコードパス**を通る。ここがずれると「手元では動くのに
凍結すると起動しない」という最悪の壊れ方をするので、テストで固定する。
"""

from __future__ import annotations

import sys

import pytest

from app import entry
from app.core.settings import Settings


class TestArgumentDispatch:
    def test_default_role_is_gui(self):
        assert entry.parse_args([]).role == "gui"

    def test_realtime_role_is_selectable(self):
        assert entry.parse_args(["--role", "realtime"]).role == "realtime"

    def test_unknown_arguments_are_passed_through(self):
        """被験者番号など、既存スクリプトの引数をそのまま渡せること。

        master_research_code.py:897 には CLI 引数処理がある。
        """
        parsed = entry.parse_args(["--role", "realtime", "--subject", "9", "--debug"])
        assert parsed.passthrough == ["--subject", "9", "--debug"]

    def test_rejects_unknown_role(self):
        with pytest.raises(SystemExit):
            entry.parse_args(["--role", "そんな役割はない"])


class TestWorkerCommand:
    def test_dev_mode_uses_module_invocation(self, monkeypatch):
        monkeypatch.setattr(entry.resources, "is_frozen", lambda: False)
        command = entry.worker_command("realtime", ["--subject", "9"])
        assert command[0] == sys.executable
        assert command[1:4] == ["-m", "app", "--role"]
        assert command[4] == "realtime"
        assert command[-2:] == ["--subject", "9"]

    def test_frozen_mode_reinvokes_the_executable(self, monkeypatch):
        """凍結時は -m app が使えない。実行ファイル自身を呼び直す。"""
        monkeypatch.setattr(entry.resources, "is_frozen", lambda: True)
        command = entry.worker_command("realtime", [])
        assert command == [sys.executable, "--role", "realtime"]

    def test_command_never_references_the_script_path(self, monkeypatch):
        """master_research_code.py を直接呼ばないこと。凍結後は存在しない。"""
        for frozen in (True, False):
            monkeypatch.setattr(entry.resources, "is_frozen", lambda: frozen)
            assert not any("master_research_code" in part for part in entry.worker_command("realtime", []))


class TestWorkerEnvironment:
    def test_settings_are_exported_into_the_environment(self):
        env = entry.worker_environment(Settings())
        assert env["DEMO_MONO_GAUGE_ON"] == "0"
        assert env["DEMO_MONO_CAM0_ONLY"] == "0"

    def test_inherits_the_parent_environment(self):
        """PATH などが消えると子プロセスが動かない。"""
        env = entry.worker_environment(Settings())
        assert "PATH" in env

    def test_overrides_win_over_inherited_values(self, monkeypatch):
        """シェルに残った古い値が勝ってはいけない。

        アプリが明示的に決めた値で子プロセスの挙動を完全に決める。
        """
        monkeypatch.setenv("DEMO_MONO_GAUGE_ON", "1")
        env = entry.worker_environment(Settings())
        assert env["DEMO_MONO_GAUGE_ON"] == "0"

    def test_marks_the_child_as_a_worker(self):
        """子プロセス側から「自分はワーカーだ」と分かるようにしておく。"""
        env = entry.worker_environment(Settings())
        assert env.get("APP_ROLE") == "realtime"


class TestQtBackend:
    def test_pyside6_is_selected(self):
        from app.core import qt

        assert qt.QT_LIB == "PySide6", (
            f"Qt バックエンドが {qt.QT_LIB}。配布物は LGPL の PySide6 である必要がある"
        )

    def test_assert_lgpl_backend_passes(self):
        from app.core import qt

        qt.assert_lgpl_backend()
