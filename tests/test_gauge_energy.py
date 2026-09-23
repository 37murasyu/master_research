"""ゲージに出す仕事量を「今のサイクルの正の仕事」にそろえることを固定する。

**なぜこのテストがあるか。**

USB 経路（``master_research_code.py``）のゲージは、手首はサイクルを検出するたびに 0 に戻るのに、
肘は起動からの累積（``_continuous_energy_J``）で一度もリセットされなかった。計測が長いほど肘の針は
上がり続け、閾値の帯（E_low・E_high）と比べる意味が無くなる（KNOWN_ISSUES §6-2 の確認の前提）。
2026-09-23 に、肘もサイクルごとにリセットすると決めた。

リセットで角度の記憶まで消すと、次のフレームで前回の角度が無くなり、1 フレーム分の仕事を落とす。
そこで仕事だけを 0 に戻し、角度は残す。

``master_research_code.py`` は import できないので、サイクル確定のブロックがリセットを呼ぶことを
AST で確かめる（``TestRealtimeWiring``）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from gauge_energy import ElbowGaugeEnergy, gauge_values

MAIN_SCRIPT = Path(__file__).resolve().parents[1] / "master_research_code.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MAIN_SCRIPT.read_text(encoding="utf-8"))


class TestElbowGaugeEnergy:
    def test_only_positive_work_is_accumulated(self):
        elbow = ElbowGaugeEnergy()
        for theta, tau in ((0.1, 2.0), (0.3, 2.0), (0.2, 2.0), (0.5, -1.0), (0.7, 1.0)):
            elbow.add("elbow_R", theta, tau)
        # +0.4（0.1→0.3）、負の仕事 −0.2 は捨てる、τ<0 で dθ>0 の −0.3 も捨てる、+0.2（0.5→0.7）
        assert elbow.value("elbow_R") == pytest.approx(0.6)
        assert elbow.value("elbow_L") == 0.0

    def test_reset_zeroes_the_work_but_keeps_the_last_angle(self):
        elbow = ElbowGaugeEnergy()
        elbow.add("elbow_R", 0.1, 2.0)
        elbow.add("elbow_R", 0.3, 2.0)
        elbow.reset_cycle()
        assert elbow.value("elbow_R") == 0.0, "サイクルが変わっても前のサイクルの仕事が残っている"
        elbow.add("elbow_R", 0.5, 2.0)
        assert elbow.value("elbow_R") == pytest.approx(0.4), "リセットで角度まで忘れ、次の増分を落とした"

    def test_a_missing_angle_skips_only_the_adjacent_steps(self):
        elbow = ElbowGaugeEnergy()
        for theta in (0.1, 0.2, np.nan, 0.4, 0.5):
            elbow.add("elbow_L", theta, 1.0)
        # 0.1→0.2 と 0.4→0.5 だけ数える（欠測をまたぐ 0.2→0.4 は数えない）
        assert elbow.value("elbow_L") == pytest.approx(0.2)


class TestGaugeValues:
    def test_each_part_uses_its_own_definition(self):
        elbow = ElbowGaugeEnergy()
        elbow.add("elbow_R", 0.0, 3.0)
        elbow.add("elbow_R", 0.5, 3.0)
        values = gauge_values(
            ["wrist_R", "elbow_R", "shoulder_R"],
            elbow,
            wrist_components={"wrist_R": [1.0, 2.0, 3.0]},
            power_history={"shoulder_R": [4.0, -1.0]},
            dt=0.5,
        )
        assert values == pytest.approx({"wrist_R": 3.0, "elbow_R": 1.5, "shoulder_R": 1.5})

    def test_parts_without_history_are_zero(self):
        values = gauge_values(["wrist_L", "shoulder_L"], ElbowGaugeEnergy(), {}, {}, dt=0.1)
        assert values == {"wrist_L": 0.0, "shoulder_L": 0.0}


def _cycle_blocks(tree: ast.AST) -> list[ast.If]:
    """``detector.update(...)`` を条件に含む if（サイクル確定のブロック）。"""
    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            calls = [n for n in ast.walk(node.test) if isinstance(n, ast.Call)]
            if any(getattr(c.func, "attr", None) == "update" and getattr(c.func.value, "id", None) == "detector"
                   for c in calls):
                blocks.append(node)
    return blocks


class TestRealtimeWiring:
    def test_the_cycle_block_resets_the_elbow_gauge(self, tree):
        """内側の条件（前回のサイクルがあり履歴が十分）の中に入れると、最初の検出でリセットされない。"""
        blocks = _cycle_blocks(tree)
        assert len(blocks) == 1, "サイクル確定のブロックが見つからない（または複数ある）"
        direct = [stmt.value for stmt in blocks[0].body if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)]
        assert any(getattr(call.func, "attr", None) == "reset_cycle" for call in direct), \
            "サイクル確定のたびに（条件なしで）肘のゲージをリセットしていない"

    def test_the_cycle_block_counts_the_cycles(self, tree):
        """ゲージの記録の cycle_index。数えないと、サイクルごとの最大が 1 つにまとまる。"""
        counted = [stmt for stmt in _cycle_blocks(tree)[0].body if isinstance(stmt, ast.AugAssign)
                   and getattr(stmt.target, "id", None) == "_gauge_cycle_index"]
        assert counted, "サイクル確定で _gauge_cycle_index を数えていない"

    def test_the_old_cumulative_counter_is_gone(self, tree):
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "_continuous_energy_J" not in names, "起動からの累積のカウンタが残っている"
        assert "ElbowGaugeEnergy" in names


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    while node in parents:
        node = parents[node]
        yield node


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def _fstring_line(tree: ast.AST, fragment: str) -> ast.JoinedStr:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
             and any(isinstance(v, ast.Constant) and fragment in str(v.value) for v in n.values)]
    assert len(found) == 1, f"{fragment} のファイル名が見つからない（または複数ある）"
    return found[0]


class TestGaugeLog:
    """HEADLESS・ファイル再生でも §6-2 を確かめられるよう、ゲージの値と閾値をファイルに残す。

    かつてゲージの値はゲージのウィンドウがあるときしか計算せず（HEADLESS では gauge=None）、
    閾値もゲージを作るときにしか計算しなかったので、画面なしでは何も確かめられなかった。
    """

    def test_values_are_computed_without_the_gauge(self, tree):
        parents = _parents(tree)
        calls = _calls(tree, "gauge_values")
        assert calls, "ゲージの値を gauge_energy.gauge_values で計算していない"
        for call in calls:
            for node in _ancestors(call, parents):
                if isinstance(node, ast.If):
                    assert not {"gauge", "HEADLESS"} & _names(node.test), \
                        f"{call.lineno} 行: ゲージがあるときしか値を計算していない"

    def test_thresholds_are_computed_without_the_gauge(self, tree):
        parents = _parents(tree)
        free = [call for call in _calls(tree, "compute_energy_thresholds")
                if not any(isinstance(node, ast.If) and {"gauge", "HEADLESS"} & _names(node.test)
                           for node in _ancestors(call, parents))]
        assert free, "閾値をゲージを作るときにしか計算していない"

    def test_the_log_is_written_after_the_cycle_debug(self, tree):
        parents = _parents(tree)
        log = _fstring_line(tree, "gauge_energy_")
        debug = _fstring_line(tree, "cycle_energy_debug_")
        assert log.lineno > debug.lineno, "ゲージの記録を cycle_energy_debug より先に書いている"
        assert not any(isinstance(node, (ast.While, ast.For)) for node in _ancestors(log, parents)), \
            "ゲージの記録をループの中で書いている（終了時に 1 回書く）"
