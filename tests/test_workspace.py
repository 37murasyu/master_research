"""凍結アプリのワークスペース（リポジトリルートの代わりになる書き込み可能な場所）を検証する。

開発時はリポジトリルートが「コードの置き場」と「データの置き場」を兼ねていて、既存スクリプトは
それを前提に書かれている:

- ``config.py`` が import 時に CWD 相対の ``output_data`` を作る
- ``calib.py`` は CWD 相対の ``camera_parameters/`` に書き、``utils.py`` は
  ``config.folder_path/camera_parameters`` から読む
- 慣性モーメント係数や 1RM 表は ``config.folder_path`` から読む

凍結すると ``folder_path`` はバンドル内（読み取り専用）になり、Finder から起動した
アプリの CWD は ``/`` になる。どちらにも書けないので、ワーカーは最初の import で落ちる。
そこで凍結時だけ、ユーザの書類フォルダにワークスペースを用意して両方の基点をそこに寄せる。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import entry
from app.core import workspace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_seed(root: Path) -> Path:
    seed = root / "seed"
    (seed / "camera_parameters").mkdir(parents=True)
    (seed / "rm_method.csv").write_text("既定の 1RM 表", encoding="utf-8")
    (seed / "camera_parameters" / "c0.dat").write_text("既定のカメラ行列", encoding="utf-8")
    return seed


class TestPrepareWorkspace:
    def test_copies_seed_files_into_an_empty_workspace(self, tmp_path):
        seed = _make_seed(tmp_path)
        ws = workspace.prepare_workspace(tmp_path / "ws", seed)

        assert (ws / "rm_method.csv").read_text(encoding="utf-8") == "既定の 1RM 表"
        assert (ws / "camera_parameters" / "c0.dat").read_text(encoding="utf-8") == "既定のカメラ行列"

    def test_never_overwrites_files_the_user_already_has(self, tmp_path):
        """キャリブレーションをやり直した結果が、起動のたびに同梱の既定値へ戻ってはいけない。"""
        seed = _make_seed(tmp_path)
        ws = tmp_path / "ws"
        (ws / "camera_parameters").mkdir(parents=True)
        (ws / "camera_parameters" / "c0.dat").write_text("校正し直した行列", encoding="utf-8")

        workspace.prepare_workspace(ws, seed)

        assert (ws / "camera_parameters" / "c0.dat").read_text(encoding="utf-8") == "校正し直した行列"

    def test_missing_seed_directory_still_yields_a_workspace(self, tmp_path):
        """同梱が漏れても、書き込み先さえあれば計測は始められる。起動は止めない。"""
        ws = workspace.prepare_workspace(tmp_path / "ws", tmp_path / "存在しない")
        assert ws.is_dir()


class TestSeedFiles:
    def test_every_seed_file_exists_in_the_repository(self):
        """spec はこの一覧から datas を組み立てる。無いファイルがあるとビルドが落ちる。"""
        missing = [rel for rel in workspace.SEED_FILES if not (REPO_ROOT / rel).is_file()]
        assert missing == []

    @pytest.mark.parametrize("file_mode", [False, True], ids=["カメラ入力", "録画入力"])
    def test_seeded_workspace_is_enough_for_the_projection_matrices(self, file_mode, tmp_path, monkeypatch):
        """初期値だけのワークスペースで、計測が使う投影行列を組み立てられること。

        録画を入力にすると ``camera_parameters/Param_for_MYvideo/`` を読む。
        これが初期値から漏れていて、凍結アプリの計測が FileNotFoundError で落ちた。
        """
        import shutil

        import utils

        seed = tmp_path / "seed"
        for rel in workspace.SEED_FILES:
            (seed / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel, seed / rel)
        ws = workspace.prepare_workspace(tmp_path / "ws", seed)
        monkeypatch.setattr(utils, "folder_path", str(ws))

        for camera_id in (0, 1):
            assert utils.get_projection_matrix(camera_id, file_mode).shape == (3, 4)


class TestRunWorkerInWorkspace:
    @pytest.fixture
    def probe(self, tmp_path, monkeypatch):
        """実行されたときの CWD と環境変数を書き出すだけのモジュール。"""
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        (mod_dir / "workspace_probe.py").write_text(
            "import os, pathlib\n"
            "pathlib.Path(os.environ['PROBE_OUT']).write_text(\n"
            "    os.getcwd() + '\\n' + os.environ.get('APP_WORKSPACE', ''), encoding='utf-8')\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(mod_dir))
        monkeypatch.setenv("PROBE_OUT", str(tmp_path / "probe.txt"))
        monkeypatch.delenv(workspace.WORKSPACE_ENV, raising=False)
        # run_worker が chdir しても、テスト後に元へ戻す
        start = tmp_path / "start"
        start.mkdir()
        monkeypatch.chdir(start)
        return tmp_path

    def _read_probe(self, tmp_path: Path) -> tuple[str, str]:
        cwd, env = (tmp_path / "probe.txt").read_text(encoding="utf-8").split("\n")
        return cwd, env

    def test_frozen_worker_runs_inside_the_workspace(self, probe, monkeypatch):
        ws = probe / "Documents" / "WheelchairTorque"
        monkeypatch.setattr(entry.resources, "is_frozen", lambda: True)
        monkeypatch.setattr(entry.resources, "resource_root", lambda: probe)
        monkeypatch.setattr(entry, "workspace_dir", lambda: ws)

        entry.run_worker("script", module="workspace_probe")

        cwd, env = self._read_probe(probe)
        assert Path(cwd).resolve() == ws.resolve()
        assert Path(env).resolve() == ws.resolve()

    def test_dev_worker_keeps_the_current_directory(self, probe, monkeypatch):
        """開発時の挙動は変えない。研究者はリポジトリルートから起動している。"""
        monkeypatch.setattr(entry.resources, "is_frozen", lambda: False)

        entry.run_worker("script", module="workspace_probe")

        cwd, env = self._read_probe(probe)
        assert Path(cwd).resolve() == (probe / "start").resolve()
        assert env == ""


class TestConfigFolderPath:
    def _folder_path(self, cwd: Path, workspace_value: str | None) -> Path:
        """別プロセスで ``config.folder_path`` を読む。config は import 時に値が決まるため。

        CWD を tmp にするのは、import 時に作られる output_data でリポジトリを汚さないため。
        """
        env = {k: v for k, v in os.environ.items() if k != workspace.WORKSPACE_ENV}
        env["PYTHONPATH"] = str(REPO_ROOT)
        if workspace_value is not None:
            env[workspace.WORKSPACE_ENV] = workspace_value
        proc = subprocess.run(
            [sys.executable, "-c", "import config; print(config.folder_path)"],
            cwd=cwd, env=env, capture_output=True, text=True, check=True,
        )
        return Path(proc.stdout.strip()).resolve()

    def test_follows_the_workspace_when_set(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert self._folder_path(tmp_path, str(ws)) == ws.resolve()

    def test_defaults_to_the_repository_root(self, tmp_path):
        assert self._folder_path(tmp_path, None) == REPO_ROOT
