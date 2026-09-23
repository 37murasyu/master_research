"""計測の CSV を、GUI が「出力先」と表示している場所に書くことを固定する。

**なぜこのテストがあるか。**

GUI は出力先を ``~/Documents/WheelchairTorque`` と表示していたが、計測（``master_research_code.py``）は
``config.save_dir`` = 作業フォルダ相対の ``output_data`` に書いていた。リポジトリから GUI を起動すると
CSV はリポジトリの ``output_data`` に出るので、GUI の停止ボタンで CSV が書かれるか（§3-2）を確かめる
とき「CSV が無い」と見誤る。2026-09-23 に、出力を表示に合わせると決めた。

- 置き場は 1 つの関数（``app.core.settings.measurement_output_dir``）から取り、表示・解析の既定の入力フォルダ・
  ワーカーへの受け渡し（環境変数 ``OUTPUT_DIR``）で共有する
- ``OUTPUT_DIR`` は GUI とワーカーの間の受け渡しで、設定画面の項目ではない（スキーマに載せない）
- GUI を通さずに起動したとき（研究者がリポジトリ直下で直接走らせる）は、従来どおり ``output_data``
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import entry
from app.core.platform_compat import user_output_dir
from app.core.settings import APP_NAME, OUTPUT_DIR_ENV, Settings, measurement_output_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _save_dir(cwd: Path, **env) -> str:
    """config を別プロセスで import して save_dir を返す（config は import 時にフォルダを作る）。"""
    environment = {k: v for k, v in os.environ.items() if k != OUTPUT_DIR_ENV}
    environment.update(env, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", "import config; print(config.save_dir)"],
                         cwd=cwd, env=environment, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


class TestWhereTheMeasurementWrites:
    def test_the_measurement_folder_is_under_the_displayed_folder(self):
        assert measurement_output_dir() == user_output_dir(APP_NAME) / "output_data"

    def test_the_worker_is_told_the_same_folder(self, monkeypatch):
        monkeypatch.setenv(OUTPUT_DIR_ENV, "/somewhere/stale")
        for role in ("realtime", "script"):
            env = entry.worker_environment(Settings(), role=role)
            assert env[OUTPUT_DIR_ENV] == str(measurement_output_dir()), "シェルに残った古い値が勝った"

    def test_config_writes_to_the_given_folder(self, tmp_path):
        target = tmp_path / "Documents" / "WheelchairTorque" / "output_data"
        assert _save_dir(tmp_path, **{OUTPUT_DIR_ENV: str(target)}) == str(target)
        assert target.is_dir(), "親フォルダが無くても作る"

    def test_without_the_gui_the_old_folder_is_used(self, tmp_path):
        assert _save_dir(tmp_path) == "output_data"
        assert (tmp_path / "output_data").is_dir()


class TestTheNameIsShared:
    def test_config_reads_the_same_variable(self):
        tree = ast.parse((REPO_ROOT / "config.py").read_text(encoding="utf-8"))
        constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
        assert OUTPUT_DIR_ENV in constants

    def test_it_is_not_a_user_setting(self):
        from tools.extract_env_schema import extract

        assert OUTPUT_DIR_ENV not in extract(REPO_ROOT / "config.py"), \
            "設定画面の項目になると、GUI が渡す値と食い違う設定を保存できてしまう"


@pytest.mark.parametrize("page", ["page_measure.py", "page_analyze.py"])
def test_the_pages_show_the_measurement_folder(page):
    source = (REPO_ROOT / "app" / "shell" / page).read_text(encoding="utf-8")
    assert "measurement_output_dir()" in source
    assert "user_output_dir(" not in source, "表示と実際の出力先を別々の式で決めている"
