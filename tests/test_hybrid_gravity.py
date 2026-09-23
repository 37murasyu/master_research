"""混成の重力の決め方（盤の短辺か体幹か）を固定する。

**なぜこのテストがあるか。**

USB 経路はチェッカーボードの短辺の向き（校正で保存）を最寄りの座標軸に吸着させて重力にし、無ければ
先頭のフレームの体幹から決める（``master_research_code.py`` の ``_pick_axis_from_vector``・
``_load_checkerboard_short_axis_runtime``・1534 行付近）。混成は体幹だけだったので、同じ選択肢を移植した。

移植で足した約束:

- 盤の短辺の符号は角点の並びで逆になりうるので、**符号は体幹で決める**（体幹の上向きと逆なら反転する）
- 盤と体幹がほぼ直交（|cos| < 0.5）なら盤の保存が壊れているとみなし、体幹に戻して理由を残す
- 盤が無ければ、今の混成と同じ ``push_up_model.estimate_gravity`` の結果（振る舞いを変えない）
- 体幹も取れなければ ``config.g``（三角測量の変換で z が上。カメラが水平という前提）
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from app.hybrid.gravity import GravityChoice, candidate_axes, choose_gravity, read_board_up
from push_up_model import estimate_gravity

G = 9.81


def _ups(vector, n=30, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vector, dtype=float)
    return np.tile(v, (n, 1)) + noise * rng.standard_normal((n, 3))


class TestReadBoardUp:
    def test_the_runtime_vector_is_used(self):
        meta = {"checkerboard_short_axis": {"vector_runtime": [0.0, 0.0, 2.0], "vector_cam0": [1.0, 0.0, 0.0]}}
        np.testing.assert_allclose(read_board_up(meta), [0.0, 0.0, 1.0])

    def test_the_camera_vector_is_converted(self):
        """vector_cam0 だけなら実行時の座標 (−x, −z, −y) に直す（calib.py の保存と同じ変換）。"""
        meta = {"checkerboard_short_axis": {"vector_cam0": [0.0, -1.0, 0.0]}}
        np.testing.assert_allclose(read_board_up(meta), [0.0, 0.0, 1.0])
        meta = {"checkerboard_short_axis": {"vector_cam0": [0.6, 0.0, 0.8]}}
        np.testing.assert_allclose(read_board_up(meta), [-0.6, -0.8, 0.0])

    @pytest.mark.parametrize(
        "meta",
        [
            None,
            {},
            {"checkerboard_short_axis": None},
            {"checkerboard_short_axis": {}},
            {"checkerboard_short_axis": {"vector_runtime": [0.0, 0.0, 0.0]}},
            {"checkerboard_short_axis": {"vector_runtime": [float("nan"), 0.0, 1.0]}},
            {"checkerboard_short_axis": {"vector_runtime": [1.0, 2.0]}},
            {"checkerboard_short_axis": {"vector_runtime": "up"}},
        ],
    )
    def test_missing_or_broken_is_none(self, meta):
        assert read_board_up(meta) is None


class TestCandidateAxes:
    def test_off_is_all_six(self):
        assert candidate_axes(False, "YZ") == ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]

    @pytest.mark.parametrize(
        "plane, expected",
        [("YZ", ["Y+", "Y-", "Z+", "Z-"]), ("XZ", ["X+", "X-", "Z+", "Z-"]), ("XY", ["X+", "X-", "Y+", "Y-"]),
         ("yz", ["Y+", "Y-", "Z+", "Z-"]), ("??", ["X+", "X-", "Y+", "Y-", "Z+", "Z-"])],
    )
    def test_a_level_plane_limits_the_axes(self, plane, expected):
        """本体の _candidate_axis_labels と同じ（知らない平面は 6 つ）。"""
        assert candidate_axes(True, plane) == expected


class TestWithoutBoard:
    @pytest.mark.parametrize("mode", ["axis", "trunk"])
    def test_same_as_estimate_gravity(self, mode):
        """盤が無ければ、今の混成（network_measure._build_inertia）と同じ重力。"""
        ups = _ups([0.15, 0.05, 0.98], noise=0.02)
        choice = choose_gravity(ups, None, magnitude=G, mode=mode)
        np.testing.assert_allclose(choice.vector, estimate_gravity(ups, G, mode).vector, rtol=0, atol=1e-12)
        assert choice.source == "trunk"
        if mode == "axis":
            assert (choice.label, choice.up_label) == ("Z-", "Z+")

    @pytest.mark.parametrize("ups", [None, np.full((30, 3), np.nan), np.zeros((0, 3))])
    def test_no_trunk_falls_back_to_the_default(self, ups):
        choice = choose_gravity(ups, None, magnitude=G)
        np.testing.assert_allclose(choice.vector, np.asarray(config.g, dtype=float))
        assert choice.source == "default" and choice.label == "Z-" and choice.detail


class TestWithBoard:
    def test_a_tilted_board_snaps_to_the_nearest_axis(self):
        """盤は少し傾いていても最寄りの軸に吸着させる（USB と同じ）。"""
        choice = choose_gravity(_ups([0.3, 0.0, 0.95]), np.array([0.1, 0.05, 0.99]), magnitude=G)
        assert isinstance(choice, GravityChoice)
        assert choice.source == "checkerboard"
        assert (choice.label, choice.up_label) == ("Z-", "Z+")
        np.testing.assert_allclose(choice.vector, [0.0, 0.0, -G])

    def test_the_sign_follows_the_trunk(self):
        """角点の並びで短辺の符号が逆でも、体幹の上向きに合わせて反転する（重力が上を向かない）。"""
        choice = choose_gravity(_ups([0.0, 0.1, 0.99]), np.array([0.05, 0.0, -0.99]), magnitude=G)
        assert choice.source == "checkerboard"
        assert choice.up_label == "Z+"
        np.testing.assert_allclose(choice.vector, [0.0, 0.0, -G])
        assert "反転" in choice.detail

    def test_the_board_decides_the_axis_even_if_the_trunk_leans(self):
        """体幹が 40° 傾いて別の軸に近くても、盤が立っていれば盤の軸を採る（盤を優先する意味）。"""
        trunk = [np.sin(np.radians(50)), 0.0, np.cos(np.radians(50))]  # 体幹だけなら X+
        assert estimate_gravity(_ups(trunk), G).up.tolist() == [1.0, 0.0, 0.0]
        choice = choose_gravity(_ups(trunk), np.array([0.0, 0.0, 1.0]), magnitude=G)
        assert choice.source == "checkerboard" and choice.up_label == "Z+"

    def test_orthogonal_board_and_trunk_fall_back_to_the_trunk(self):
        """盤と体幹がほぼ直交なら、盤の保存が壊れているとみなして体幹に戻し、理由を残す。"""
        ups = _ups([0.0, 0.0, 1.0])
        choice = choose_gravity(ups, np.array([1.0, 0.0, 0.1]), magnitude=G)
        assert choice.source == "trunk"
        np.testing.assert_allclose(choice.vector, estimate_gravity(ups, G).vector)
        assert "直交" in choice.detail

    def test_without_a_trunk_the_board_is_used_as_is(self):
        choice = choose_gravity(None, np.array([0.0, 0.0, 1.0]), magnitude=G)
        assert choice.source == "checkerboard" and choice.up_label == "Z+"
        assert "体幹" in choice.detail

    def test_the_level_plane_limits_the_snap(self):
        """水平面の制約（GRAVITY_LEVEL_PLANE）があれば、吸着先を候補の軸に限る。"""
        board = np.array([0.6, 0.1, 0.55])  # 6 つからなら X+
        choice = choose_gravity(_ups([0.3, 0.0, 0.95]), board, magnitude=G, candidates=candidate_axes(True, "YZ"))
        assert choice.source == "checkerboard" and choice.up_label == "Z+"

    def test_the_preferred_axis_wins_when_ambiguous(self):
        """上位 2 軸の cos の差が ambiguity 未満なら、preferred_gravity の逆（上向き）を選ぶ（本体と同じ）。"""
        board = np.array([0.714, 0.0, 0.70])
        ups = _ups([0.5, 0.0, 0.87])
        assert choose_gravity(ups, board, magnitude=G).up_label == "X+"
        choice = choose_gravity(ups, board, magnitude=G, preferred_gravity="Z-", ambiguity=0.08)
        assert choice.up_label == "Z+" and choice.label == "Z-"

    def test_a_broken_board_vector_is_ignored(self):
        ups = _ups([0.0, 0.0, 1.0])
        choice = choose_gravity(ups, np.array([np.nan, 0.0, 1.0]), magnitude=G)
        assert choice.source == "trunk"

    def test_the_magnitude_is_used(self):
        choice = choose_gravity(_ups([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), magnitude=9.80665)
        np.testing.assert_allclose(choice.vector, [0.0, 0.0, -9.80665])
