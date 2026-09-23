"""混成のゲージの閾値（論文 4.5.2 節の W_0.70・W_0.85）を固定する。

**なぜこのテストがあるか。**

ゲージの帯は論文に則る（ユーザーの指示）。FB 尺度の閾値は負荷率 c = 0.70・0.85 の理論仕事
W_c = theoretical_1rm_work(joint, 体重, 前腕長, c·1RM) で、スコアでは S ≈ 0.72・0.86 に当たる（表 2）。
USB 経路のゲージは旧来の式（r_x 直書き、体重が 2 系統）のままで §2-6 の修正に追随していないので、
混成では使わずにこちらで作り直した。ここでは次を固定する:

- 被験者番号は "00"・"0"・" 03 " のどれでも整数として表を引く（表の subject_id は整数）
- 1RM の列はオフラインのスコア（``ONE_RM_COLUMNS``）と同じ: 肘は ``elbow_{side}_outer``、手首は ``wrist_{side}``
- 帯の数値（被験者 00・65 kg・前腕 0.25 m の見積もり 47.47/56.89/66.31 J、9.65/11.58/13.52 J）
- 前腕長が人体の範囲の外・1RM が無いときは帯を出さず、理由を残す（黙って 0 J の帯を出さない）
- 状態の境界: v < lo 不足、lo ≤ v < hi 目標帯、v ≥ hi 過負荷（表 2 は 0.86 ≦ S が最大筋力）
"""

from __future__ import annotations

import math
import os

import pytest

from app.gauge import thresholds
from app.gauge.thresholds import PartBand, classify, load_one_rm, part_bands, subject_index
from compute_cycle_energy_elbow_wrist import theoretical_1rm_work

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONE_RM_TABLE = os.path.join(REPO_ROOT, "m_max_all_merged.csv")

HEADER = "subject_id,wrist_L,wrist_R,elbow_L,elbow_R,wrist_L_inner,wrist_R_inner,elbow_L_outer,elbow_R_outer\n"


def _table(tmp_path, *rows: str):
    path = tmp_path / "m_max.csv"
    path.write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


class TestSubjectIndex:
    @pytest.mark.parametrize("raw, expected", [("00", 0), ("0", 0), ("7", 7), (" 03 ", 3), ("21", 21)])
    def test_digits_are_read_as_an_integer(self, raw, expected):
        """GUI の SUBJECT_ID は文字列（例 "00"）。表の subject_id は整数なので "00" と 0 を同じに読む。"""
        assert subject_index(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "  ", "S000", "abc", "-1", "1.5"])
    def test_anything_else_is_none(self, raw):
        """数字でなければ引かない。理由のログは呼び出し側が出す（ここで黙って 0 にしない）。"""
        assert subject_index(raw) is None


class TestLoadOneRm:
    def test_the_repository_table_has_subject_zero(self):
        """朝の実機は被験者 00。追跡している表から肘 outer 15 kg・手首 8.235 kg が引けること。"""
        one_rm = load_one_rm(ONE_RM_TABLE, 0)
        assert one_rm == {"elbow_L": 15.0, "elbow_R": 15.0, "wrist_L": 8.235, "wrist_R": 8.235}

    def test_columns_follow_the_offline_score(self, tmp_path):
        """肘は elbow_{side}_outer（伸展の力、§6-4）、手首は wrist_{side}。inner や素の elbow を取り違えない。"""
        path = _table(tmp_path, "5,1,2,3,4,5,6,7,8")
        assert load_one_rm(path, 5) == {"elbow_L": 7.0, "elbow_R": 8.0, "wrist_L": 1.0, "wrist_R": 2.0}

    def test_none_and_blank_are_missing(self, tmp_path):
        """表には "none" と空欄がある（被験者 21〜25）。その部位だけ None にして他は使う。"""
        path = _table(tmp_path, "22,8.75,nan,15,14,14,13,none,")
        assert load_one_rm(path, 22) == {"elbow_L": None, "elbow_R": None, "wrist_L": 8.75, "wrist_R": None}

    def test_an_unknown_subject_is_all_none(self, tmp_path):
        path = _table(tmp_path, "1,1,1,1,1,1,1,1,1")
        assert load_one_rm(path, 99) == {"elbow_L": None, "elbow_R": None, "wrist_L": None, "wrist_R": None}

    def test_subject_ids_written_with_zeros_still_match(self, tmp_path):
        """表の番号が "00" と書かれていても整数として突き合わせる。"""
        path = _table(tmp_path, "00,1,2,3,4,5,6,7,8")
        assert load_one_rm(path, 0)["elbow_L"] == 7.0


class TestPartBands:
    ONE_RM = {"elbow_L": 15.0, "elbow_R": 15.0, "wrist_L": 8.235, "wrist_R": 8.235}

    def test_the_band_is_the_theoretical_work_at_70_and_85_percent(self):
        """論文 4.5.2 節: W_c = theoretical_1rm_work(joint, M, L, c·1RM)、c = 0.70・0.85。"""
        bands = part_bands(65.0, {"L": 0.25, "R": 0.25}, self.ONE_RM)
        for part, joint, m in (("elbow_L", "elbow", 15.0), ("wrist_R", "wrist", 8.235)):
            b = bands[part]
            assert b.band is not None
            assert b.band[0] == pytest.approx(theoretical_1rm_work(joint, 65.0, 0.25, 0.70 * m), rel=1e-9)
            assert b.band[1] == pytest.approx(theoretical_1rm_work(joint, 65.0, 0.25, 0.85 * m), rel=1e-9)
            assert b.w1rm == pytest.approx(theoretical_1rm_work(joint, 65.0, 0.25, m), rel=1e-9)
            assert b.one_rm_kg == m and b.forearm_m == 0.25 and b.reason == ""

    def test_the_estimate_for_subject_zero(self):
        """計画の見積もり（朝にユーザーへ伝える数値）。見積もりは小数 2 桁なので ±0.01 J で確かめる。"""
        bands = part_bands(65.0, {"L": 0.25, "R": 0.25}, self.ONE_RM)
        elbow, wrist = bands["elbow_L"], bands["wrist_L"]
        assert (*elbow.band, elbow.w1rm) == pytest.approx((47.47, 56.89, 66.31), abs=0.01)
        assert (*wrist.band, wrist.w1rm) == pytest.approx((9.65, 11.58, 13.52), abs=0.01)

    def test_each_side_uses_its_own_forearm(self):
        bands = part_bands(65.0, {"L": 0.24, "R": 0.28}, self.ONE_RM)
        assert bands["elbow_L"].forearm_m == 0.24
        assert bands["elbow_R"].forearm_m == 0.28
        assert bands["elbow_R"].band[0] > bands["elbow_L"].band[0]

    @pytest.mark.parametrize("forearm", [0.10, 0.45, float("nan"), None])
    def test_an_implausible_forearm_gives_no_band(self, forearm):
        """前腕長は三角測量の中央値。0.15〜0.40 m の外は 3D が壊れているので、帯を出さず理由を残す。"""
        bands = part_bands(65.0, {"L": forearm, "R": 0.25}, self.ONE_RM)
        for part in ("elbow_L", "wrist_L"):
            assert bands[part].band is None and bands[part].w1rm is None
            assert "前腕" in bands[part].reason
        assert bands["elbow_R"].band is not None

    def test_a_missing_one_rm_gives_no_band(self):
        one_rm = dict(self.ONE_RM, wrist_R=None)
        bands = part_bands(65.0, {"L": 0.25, "R": 0.25}, one_rm)
        assert bands["wrist_R"].band is None and bands["wrist_R"].w1rm is None
        assert "1RM" in bands["wrist_R"].reason
        assert bands["wrist_L"].band is not None

    def test_a_missing_side_in_the_forearm_map_gives_no_band(self):
        bands = part_bands(65.0, {"L": 0.25}, self.ONE_RM)
        assert bands["elbow_R"].band is None and "前腕" in bands["elbow_R"].reason

    def test_all_four_parts_are_always_present(self):
        bands = part_bands(65.0, {}, {})
        assert set(bands) == set(thresholds.PARTS)
        assert all(isinstance(b, PartBand) for b in bands.values())

    def test_a_part_band_is_frozen(self):
        band = part_bands(65.0, {"L": 0.25, "R": 0.25}, self.ONE_RM)["elbow_L"]
        with pytest.raises(Exception):
            band.w1rm = 0.0  # type: ignore[misc]


class TestClassify:
    BAND = (47.47, 56.89)

    @pytest.mark.parametrize(
        "value, expected",
        [(0.0, "under"), (47.46, "under"), (47.47, "target"), (56.88, "target"), (56.89, "over"), (100.0, "over")],
    )
    def test_boundaries(self, value, expected):
        """lo ちょうどは目標帯、hi ちょうどは過負荷（表 2 は 0.86 ≦ S が最大筋力）。"""
        assert classify(value, self.BAND) == expected

    def test_no_band_means_no_state(self):
        assert classify(50.0, None) is None

    def test_a_missing_value_means_no_state(self):
        assert classify(math.nan, self.BAND) is None
        assert classify(None, self.BAND) is None
