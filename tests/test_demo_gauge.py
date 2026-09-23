"""混成のデモのゲージ（DemoGauge）を固定する。

**なぜこのテストがあるか。**

USB の単眼デモ（``master_research_code.py`` の ``DEMO_MONO_GAUGE_ON``）は既定では動いていなかった。
肘角は 2D の古い並び順の添字（右 2,1,0・左 3,4,5）で引いており、肩は Tasks 経路に world landmarks が
無いので常に None だった。混成では三角測量した 3D で作り直した:

- 肩の上昇 = 肩の中点の「重力の上向き」への射影の、基準（EMA 0.01）からの差
- 肘角の変化 = 3D の肘角（肩−肘と手首−肘のなす角）の、基準（EMA 0.01）からの差の絶対値
- 段階の目標比: 肩 ≥ 0.10 m かつ肘 ≥ 45° → 0.80、肩 ≥ 0.02 m かつ肘 ≥ 8° → 0.30、それ以外 0
- 1 フレームごとに +0.025 / −0.035 で目標へ近づける。同じ側の elbow と wrist に同じ比
- 比を J に直すとき、比 0.80 を帯の中央（(lo+hi)/2）に当てる（帯が無ければ 0.80 → 50 J）
"""

from __future__ import annotations

import numpy as np
import pytest

from app.gauge.thresholds import PartBand, part_bands
from app.hybrid.demo_gauge import DemoConfig, DemoGauge
from config import pose_keypoints, slot_of

UP = np.array([0.0, 0.0, 1.0])
N_POINTS = len(pose_keypoints)


def _bands():
    return part_bands(
        65.0, {"L": 0.25, "R": 0.25}, {"elbow_L": 15.0, "elbow_R": 15.0, "wrist_L": 8.235, "wrist_R": 8.235}
    )


def _body(lift: float = 0.0, elbow_deg=(90.0, 90.0)) -> np.ndarray:
    """z が上の座った姿勢。肩を lift [m] 上げ、左右の肘角を elbow_deg にする（上腕は鉛直、前腕は前へ）。"""
    pts = np.zeros((N_POINTS, 3))
    for side, x, angle in (("L", 0.2, elbow_deg[0]), ("R", -0.2, elbow_deg[1])):
        shoulder = np.array([x, 0.0, 1.0 + lift])
        elbow = shoulder - np.array([0.0, 0.0, 0.3])
        a = np.radians(angle)
        # 肘角 = 上腕の向き（肘→肩 = +z）と前腕（肘→手首）のなす角
        wrist = elbow + 0.25 * np.array([0.0, np.sin(a), np.cos(a)])
        pts[slot_of(f"{side}_SHOULDER")] = shoulder
        pts[slot_of(f"{side}_ELBOW")] = elbow
        pts[slot_of(f"{side}_WRIST")] = wrist
        pts[slot_of(f"{side}_HIP")] = np.array([x * 0.8, 0.0, 0.5])
    return pts


def _feed(gauge, frames, bands=None, up=UP):
    out = None
    for pts in frames:
        out = gauge.update(pts, up, bands)
    return out


class TestStages:
    def test_sitting_still_is_zero(self):
        gauge = DemoGauge()
        values = _feed(gauge, [_body()] * 60, _bands())
        assert values == {"elbow_L": 0.0, "elbow_R": 0.0, "wrist_L": 0.0, "wrist_R": 0.0}

    def test_a_full_push_reaches_the_middle_of_the_band(self):
        """肩 15 cm・肘 80° の押し上げで比は 0.80 まで上がり、J は帯の中央（目標帯の中）に当たる。"""
        gauge = DemoGauge()
        bands = _bands()
        _feed(gauge, [_body()] * 30, bands)
        values = _feed(gauge, [_body(0.15, (170.0, 170.0))] * 32, bands)
        for part in ("elbow_L", "elbow_R", "wrist_L", "wrist_R"):
            lo, hi = bands[part].band
            assert values[part] == pytest.approx((lo + hi) / 2)
        assert gauge.ratios == pytest.approx({"L": 0.80, "R": 0.80})

    def test_the_ratio_moves_by_steps(self):
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30)
        gauge.update(_body(0.15, (170.0, 170.0)), UP, None)
        assert gauge.ratios["L"] == pytest.approx(0.025)
        _feed(gauge, [_body(0.15, (170.0, 170.0))] * 40)
        top = gauge.ratios["L"]
        gauge.update(_body(), UP, None)
        assert gauge.ratios["L"] == pytest.approx(max(0.0, top - 0.035))

    def test_a_partial_push_targets_030(self):
        """肩 4 cm・肘 15° は部分（0.30）。帯が無い部位は 0.80 → 50 J の固定の目盛り。"""
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30)
        values = _feed(gauge, [_body(0.04, (105.0, 105.0))] * 20)
        assert gauge.ratios["L"] == pytest.approx(0.30)
        assert values["elbow_L"] == pytest.approx(0.30 / 0.80 * 50.0)

    def test_both_conditions_are_needed(self):
        """肩だけ上がって肘が変わらない（体を傾けただけ）なら上がらない。"""
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30)
        _feed(gauge, [_body(0.15, (90.0, 90.0))] * 20)
        assert gauge.ratios == {"L": 0.0, "R": 0.0}

    def test_each_side_uses_its_own_elbow(self):
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30)
        values = _feed(gauge, [_body(0.15, (170.0, 90.0))] * 20)
        assert values["elbow_L"] > 0.0 and values["wrist_L"] > 0.0
        assert values["elbow_R"] == 0.0 and values["wrist_R"] == 0.0
        assert gauge.ratios["L"] > 0.0 and gauge.ratios["R"] == 0.0

    def test_elbow_and_wrist_share_the_ratio_but_not_the_scale(self):
        """同じ側の elbow と wrist は同じ比。J は部位ごとの帯の中央で目盛る。"""
        gauge = DemoGauge()
        bands = _bands()
        _feed(gauge, [_body()] * 30, bands)
        values = _feed(gauge, [_body(0.15, (170.0, 170.0))] * 10, bands)
        mid = {p: sum(bands[p].band) / 2 for p in bands}
        assert values["elbow_L"] / mid["elbow_L"] == pytest.approx(values["wrist_L"] / mid["wrist_L"])

    def test_the_baseline_follows_slowly(self):
        """基準は EMA 0.01 で追う（USB と同じ）。持ち上げたまま止まると、段階はやがて下がる。"""
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30)
        _feed(gauge, [_body(0.15, (170.0, 170.0))] * 300)
        assert gauge.ratios["L"] < 0.80


class TestRobustness:
    def test_missing_points_decay_to_zero(self):
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30)
        _feed(gauge, [_body(0.15, (170.0, 170.0))] * 32)
        values = _feed(gauge, [np.full((N_POINTS, 3), np.nan)] * 30)
        assert values == {"elbow_L": 0.0, "elbow_R": 0.0, "wrist_L": 0.0, "wrist_R": 0.0}

    def test_nan_does_not_poison_the_baseline(self):
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 5)
        _feed(gauge, [np.full((N_POINTS, 3), np.nan)] * 5)
        values = _feed(gauge, [_body(0.15, (170.0, 170.0))] * 32)
        assert values["elbow_L"] == pytest.approx(50.0)

    @pytest.mark.parametrize("up", [None, np.zeros(3), np.array([np.nan, 0, 1])])
    def test_no_up_direction_means_no_shoulder_rise(self, up):
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 10, up=up)
        _feed(gauge, [_body(0.15, (170.0, 170.0))] * 10, up=up)
        assert gauge.ratios == {"L": 0.0, "R": 0.0}

    def test_the_up_vector_is_normalised(self):
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30, up=UP * 9.81)
        _feed(gauge, [_body(0.15, (170.0, 170.0))] * 32, up=UP * 9.81)
        assert gauge.ratios["L"] == pytest.approx(0.80)

    def test_a_band_without_a_range_uses_the_fixed_scale(self):
        bands = dict(_bands())
        bands["wrist_R"] = PartBand(None, None, None, 0.5, "前腕長 0.500 m が範囲の外")
        gauge = DemoGauge()
        _feed(gauge, [_body()] * 30, bands)
        values = _feed(gauge, [_body(0.15, (170.0, 170.0))] * 32, bands)
        assert values["wrist_R"] == pytest.approx(50.0)


class TestConfig:
    def test_defaults_match_the_usb_demo(self):
        c = DemoConfig()
        assert (c.shoulder_full_m, c.elbow_full_deg, c.shoulder_partial_m, c.elbow_partial_deg) == (0.10, 45.0, 0.02, 8.0)
        assert (c.ratio_full, c.ratio_partial, c.up_step, c.down_step, c.baseline_ema) == (0.80, 0.30, 0.025, 0.035, 0.01)
        assert c.unbanded_full_j == 50.0

    def test_from_env_reads_the_usb_names(self):
        c = DemoConfig.from_env({"DEMO_SHOULDER_RISE_FULL_M": "0.08", "DEMO_RATIO_UP_STEP": "0.05", "DEMO_BASELINE_EMA": "x"})
        assert c.shoulder_full_m == 0.08 and c.up_step == 0.05 and c.baseline_ema == 0.01
