"""環境変数の真偽値の読み方（``config.env_flag``）と、設定スキーマへの取り込みを固定する。

**なぜこのテストがあるか。**

``master_research_code.py`` は真偽値を ``os.getenv(X, '1') in ('1','true','True')`` と
``os.getenv(X, '1') not in ('0','false','False')`` の 2 通りで読んでいた（KNOWN_ISSUES §4-4）。
2 つは ``''``・``'yes'``・``'TRUE'`` などで結果が逆になる。しかもスキーマの抽出器
（``tools/extract_env_schema.py``）は後者を bool と認識せず str 扱いにしており、GUI では
チェックボックスでなくテキスト欄になっていた（``DEBUG_LOGS``・``GRAB_PARALLEL`` など）。
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from config import env_flag

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestEnvFlag:
    @pytest.mark.parametrize("raw", ["1", "true", "True", "TRUE", "yes", "on", " 1 "])
    def test_truthy_values(self, monkeypatch, raw):
        monkeypatch.setenv("WT_FLAG", raw)
        assert env_flag("WT_FLAG", False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "False", "no", "OFF"])
    def test_falsy_values(self, monkeypatch, raw):
        monkeypatch.setenv("WT_FLAG", raw)
        assert env_flag("WT_FLAG", True) is False

    @pytest.mark.parametrize("default", [True, False])
    def test_unset_or_unknown_values_fall_back_to_the_default(self, monkeypatch, default):
        monkeypatch.delenv("WT_FLAG", raising=False)
        assert env_flag("WT_FLAG", default) is default
        for raw in ("", "maybe"):
            monkeypatch.setenv("WT_FLAG", raw)
            assert env_flag("WT_FLAG", default) is default


class TestSchemaExtraction:
    def test_env_flag_calls_become_bool_settings(self, tmp_path):
        from tools.extract_env_schema import extract

        source = tmp_path / "script.py"
        source.write_text(textwrap.dedent("""
            from config import env_flag
            A = env_flag('WT_ON', True)
            B = env_flag("WT_OFF", False)
        """), encoding="utf-8")
        found = extract(source)
        assert found["WT_ON"]["type"] == "bool" and found["WT_ON"]["default"] == "1"
        assert found["WT_OFF"]["type"] == "bool" and found["WT_OFF"]["default"] == "0"


class TestMainScriptUsesEnvFlag:
    """真偽値の慣用句 2 通りが env_flag に置き換わっている。"""

    def test_no_literal_tuple_comparisons_remain(self):
        tree = ast.parse((REPO_ROOT / "master_research_code.py").read_text(encoding="utf-8"))
        leftovers = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Call)):
                continue
            func = node.left.func
            if not (isinstance(func, ast.Attribute) and func.attr == "getenv"):
                continue
            comparator = node.comparators[0]
            if isinstance(comparator, ast.Tuple):
                values = [e.value for e in comparator.elts if isinstance(e, ast.Constant)]
                if values in (["1", "true", "True"], ["0", "false", "False"]):
                    leftovers.append(node.lineno)
        assert not leftovers, f"os.getenv の真偽値の慣用句が残っている（行 {leftovers}）"


@pytest.mark.parametrize("name,default", [
    ("USE_POSE_LANDMARKER", True), ("USE_NATIVE_POSE", False),
    ("GAUGE_THRESH_AUTO", True), ("IMMEDIATE_ESC_BREAK", True),
    ("PERF_LOG", False), ("LOOP_FILE_PLAYBACK", False), ("PERF_TRACE", False),
    ("E_FC_ADAPTIVE_ON", False),
])
def test_remaining_flags_use_shared_parser_and_bool_schema(name, default):
    """残った真偽値設定も TRUE/on を受け付け、GUI に真偽値として出すため。"""
    from tools.extract_env_schema import extract

    path = REPO_ROOT / "master_research_code.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "env_flag" and len(node.args) == 2
             and isinstance(node.args[0], ast.Constant) and node.args[0].value == name]
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[1]) is default
    setting = extract(path)[name]
    assert setting["type"] == "bool"
    assert setting["default"] == str(int(default))
