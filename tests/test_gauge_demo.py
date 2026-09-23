"""合成データのデモ（``app.gauge.demo``）を確かめる。

CLI には 3 つの使い方があるが、引数なしの経路（``GaugeWindow`` を開いて
``QTimer`` で実時間 2 秒待ってから流し始める）は目で見て確かめる以外に
自動化しづらいので、ここでは対象にしない。試験にするのは:

1. ``SCENARIOS``（``--snapshot``・目で見ての確認の両方が使う土台）が
   task-9-brief.md の 11 個の名前をそのまま持ち、局面や null・過負荷の張り付き・
   replay の印など、状態の描き分けを一通り踏んでいること。
2. ``--snapshot DIR`` が状態ごとに 1 枚ずつ PNG を書くこと。
3. ``--emit`` が ``@@GAUGE `` の行だけを標準出力へ書き、終了コードを選べること。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.gauge import demo
from app.gauge import model as gm
from app.gauge.protocol import PART_NAMES, decode

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCENARIO_NAMES = [
    "01_waiting",
    "02_first",
    "03_nth",
    "04_joules_off",
    "05_band_null",
    "06_over_clamp",
    "07_done",
    "08_failed",
    "09_wide_band",
    "10_replay",
    "11_now_null",
]


class TestScenarios:
    def test_scenario_covers_every_state(self):
        """名前は brief のまま・順序どおり。局面（Phase）を 4 つとも、少なくとも
        1 つのシナリオが持つ。
        """
        assert list(demo.SCENARIOS.keys()) == _SCENARIO_NAMES
        phases = {state.phase for state in demo.SCENARIOS.values()}
        assert phases == set(gm.Phase)

    def test_04_joules_off_is_the_only_scenario_with_joules_off(self):
        assert demo.SCENARIOS["04_joules_off"].show_joules is False
        others = [s for name, s in demo.SCENARIOS.items() if name != "04_joules_off"]
        assert all(s.show_joules for s in others)

    def test_05_band_null_has_exactly_one_part_without_a_band(self):
        parts = demo.SCENARIOS["05_band_null"].frame.parts
        nulled = [p for p in PART_NAMES if parts[p].band is None]
        assert nulled == ["wrist_R"]

    def test_06_over_clamp_has_a_part_pinned_at_the_right_edge(self):
        """1.25·hi を超えた部位は ``fraction`` が 1.0 に張り付く（constraints.md）。"""
        parts = demo.SCENARIOS["06_over_clamp"].frame.parts
        assert any(gm.fraction(r.now, r.band) == 1.0 for r in parts.values())

    def test_09_wide_band_uses_the_stated_50_to_200_band(self):
        parts = demo.SCENARIOS["09_wide_band"].frame.parts
        assert all(r.band == (50.0, 200.0) for r in parts.values())

    def test_10_replay_is_the_only_scenario_with_the_replay_pill(self):
        """``model.header`` の ``replay`` は ``source == "replay"`` のときだけ真。"""
        for name, state in demo.SCENARIOS.items():
            assert gm.header(state).replay == (name == "10_replay"), name

    def test_11_now_null_has_exactly_one_part_without_a_value(self):
        parts = demo.SCENARIOS["11_now_null"].frame.parts
        nulled = [p for p in PART_NAMES if parts[p].now is None]
        assert nulled == ["wrist_R"]
        assert parts["wrist_R"].band is not None  # 帯はある（今回値だけ無い）


class TestSnapshot:
    def test_snapshot_writes_one_png_per_state(self, tmp_path):
        pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")
        from app.core.qt import QtGui

        code = demo.main(["--snapshot", str(tmp_path), "--size", "320x180"])
        assert code == 0

        written = sorted(p.name for p in tmp_path.glob("*.png"))
        assert written == sorted(f"{name}.png" for name in demo.SCENARIOS)

        for name in demo.SCENARIOS:
            image = QtGui.QImage(str(tmp_path / f"{name}.png"))
            assert not image.isNull(), name
            assert (image.width(), image.height()) == (320, 180), name


class TestEmit:
    def test_emit_prints_decodable_lines(self):
        """別プロセスで --emit --count 5 --interval 0 を走らせ、5 行すべてが
        decode でき、ほかの行が無いことを確かめる。
        """
        result = subprocess.run(
            [sys.executable, "-m", "app.gauge.demo", "--emit", "--count", "5", "--interval", "0"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        assert len(lines) == 5
        for line in lines:
            frame = decode(line)
            assert frame is not None, line
            assert frame.source == "demo"

    def test_emit_exit_code_option(self):
        result = subprocess.run(
            [sys.executable, "-m", "app.gauge.demo", "--emit", "--count", "1", "--interval", "0", "--exit-code", "3"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 3
