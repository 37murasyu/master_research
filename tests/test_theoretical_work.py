"""スコアの分母（理論 1RM 仕事量）の定義を固定する。

**なぜこのテストがあるか。**

手首の分母はダンベルのてこの腕を 0 としていた（論文 53 ページの定義）。手首の 1RM が分母に効かず、
分母は手の重さの項だけの 0.6〜2 J になり、手首のスコアが 2〜30 というありえない値になっていた
（KNOWN_ISSUES §2-6）。2026-09-23 に定義を直すと決めた: リストカールのダンベルは手のひらにあり、
手首の軸から手の中心（手長 × 0.506）だけ離れている。手長は実測が無いので、前腕長から
Drillis & Contini の体節長比（手 0.108H、前腕 0.146H）で推定する。

肘の分母（前腕＋手、ダンベルは前腕長の位置）は変えない。1RM は ``elbow_*_outer``
（肘が内側へ曲がろうとするのを打ち消す向きの筋力、つまり伸展の力。§6-4）を使う。
"""

from __future__ import annotations

import pytest

from compute_cycle_energy_elbow_wrist import theoretical_1rm_work
from config import THEORETICAL_WORK_COEFF as K

BODY = 60.0
FOREARM = 0.25


class TestWrist:
    def test_the_dumbbell_acts_at_the_centre_of_the_hand(self):
        hand = FOREARM * 0.108 / 0.146
        lever = 0.506 * hand
        expected = (BODY * 0.006 * lever + 5.0 * lever) * K
        assert theoretical_1rm_work("wrist", BODY, FOREARM, 5.0) == pytest.approx(expected)

    def test_the_one_rm_now_counts(self):
        light = theoretical_1rm_work("wrist", BODY, FOREARM, 5.0)
        heavy = theoretical_1rm_work("wrist", BODY, FOREARM, 10.0)
        assert heavy > 1.8 * light, "手首の 1RM が分母に効いていない（てこの腕が 0 のまま）"


class TestElbow:
    def test_the_elbow_definition_is_unchanged(self):
        m_forearm, m_hand = BODY * 0.016, BODY * 0.006
        r_g = (m_forearm * 0.430 * FOREARM + m_hand * FOREARM) / (m_forearm + m_hand)
        expected = ((m_forearm + m_hand) * r_g + 10.0 * FOREARM) * K
        assert theoretical_1rm_work("elbow", BODY, FOREARM, 10.0) == pytest.approx(expected)


def test_an_unknown_joint_is_an_error():
    with pytest.raises(ValueError):
        theoretical_1rm_work("shoulder", BODY, FOREARM, 1.0)
