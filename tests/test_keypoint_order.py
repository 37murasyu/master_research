"""キーポイントの並び順を固定する。

**なぜこのテストがあるか。**

``config.pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, ...]`` の並びには意味が無い。
元実装 TemugeB/bodypose3d は::

    for i, landmark in enumerate(results.pose_landmarks.landmark):
        if i not in pose_keypoints: continue

とランドマーク ID の**昇順**に走査しており、このリストは「どの点を使うか」の
フィルタとしてしか働かない。ところがこのリポジトリの抽出はリストの並び順で
回しており、3D 点列が

    [0]=右手首 [1]=右肘 [2]=右肩 [3]=左肩 [4]=左肘 [5]=左手首 ...

という別の並びになっていた。``master_research_code.py`` の ``part_calculations`` は
昇順を前提にしているため、**8 リンク中 7 本が体を斜めに横切るベクトル**になり、
慣性テンソルが 4.5〜7.8 倍過大、上胴体は負になっていた（再検算 R-1）。

抽出側を昇順に直したので、その並びをここで固定する。

.. note::
   ``master_research_code.py`` はトップレベルにスクリプト本体（カメラ初期化を含む）を
   持つため import できない。必要な定義だけ AST で取り出して検証する。
"""

from __future__ import annotations

import ast
import io
import os

import numpy as np
import pytest

from config import pose_keypoints

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SCRIPT = os.path.join(REPO_ROOT, "master_research_code.py")

# 抽出はランドマーク ID の昇順で返す。期待値は config から導出する
# （値を写すと config 側の変更を検知できない）。
EXPECTED_ORDER = sorted(pose_keypoints)

LANDMARK_NAME = {
    11: "左肩", 12: "右肩", 13: "左肘", 14: "右肘", 15: "左手首", 16: "右手首",
    17: "左小指", 18: "右小指", 19: "左人差指", 20: "右人差指",
    21: "左親指", 22: "右親指", 23: "左腰", 24: "右腰",
    25: "左膝", 26: "右膝", 27: "左足首", 28: "右足首",
}


def _slot_name(index: int, keypoints=None) -> str:
    """位置索引から部位名を引く。構成が変われば対応も変わる。"""
    ids = sorted(pose_keypoints if keypoints is None else keypoints)
    return LANDMARK_NAME[ids[index]]


def _load_from_script(name: str, kind: str):
    """スクリプトを実行せずに、トップレベルの定義を 1 つだけ取り出す。

    ``kind`` は "assign"（リテラル代入）か "function"（関数定義）。
    """
    tree = ast.parse(io.open(MAIN_SCRIPT, encoding="utf-8").read())
    for node in tree.body:
        if kind == "assign" and isinstance(node, ast.Assign):
            if any(getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
        if kind == "function" and isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict = {}
            exec(compile(module, MAIN_SCRIPT, "exec"), namespace)  # noqa: S102
            return namespace[name]
    raise AssertionError(f"{MAIN_SCRIPT} に {kind} {name} が見つからない")


class _FakeLandmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeResults:
    """MediaPipe の results を模す。ランドマーク ID が座標に出るようにしておく。"""

    def __init__(self, count: int = 33):
        class _Holder:
            pass

        holder = _Holder()
        # x に ID を入れておけば、戻り値からどの ID が来たか判別できる
        holder.landmark = [_FakeLandmark(pid / 1000.0, pid / 1000.0) for pid in range(count)]
        self.pose_landmarks = holder
        self.pose_world_landmarks = holder


def _decode_ids(pixels, width: int) -> list[int]:
    """抽出結果のピクセル座標から、元のランドマーク ID を復元する。"""
    return [int(round(px / width * 1000.0)) for px, _ in pixels]


class TestExtractKeypointsOrder:
    """抽出はランドマーク ID の昇順で返す。"""

    WIDTH = 1000

    def _frame(self):
        return np.zeros((self.WIDTH, self.WIDTH, 3), dtype=np.uint8)

    def test_returns_landmark_ids_in_ascending_order(self):
        from utils import extract_keypoints

        got, _ = extract_keypoints(
            _FakeResults(), _FakeResults(), list(pose_keypoints), self._frame(), self._frame())
        assert _decode_ids(got, self.WIDTH) == EXPECTED_ORDER, (
            "抽出がランドマーク ID の昇順になっていない。"
            " pose_keypoints の並び順で回していないか確認すること"
        )

    def test_order_is_independent_of_the_declaration_order(self):
        """``pose_keypoints`` の並びを入れ替えても結果が変わらない。"""
        from utils import extract_keypoints

        shuffled = list(reversed(list(pose_keypoints)))
        got, _ = extract_keypoints(
            _FakeResults(), _FakeResults(), shuffled, self._frame(), self._frame())
        assert _decode_ids(got, self.WIDTH) == EXPECTED_ORDER, (
            "宣言順を変えたら抽出順も変わった。sorted() を通していない"
        )

    def test_fast_path_agrees_with_the_reference_path(self):
        """高速版 ``_extract_keypoints_fast_single`` も同じ順で返す。"""
        fast = _load_from_script("_extract_keypoints_fast_single", "function")
        got = fast(_FakeResults(), list(pose_keypoints), (self.WIDTH, self.WIDTH, 3))
        assert _decode_ids(got, self.WIDTH) == EXPECTED_ORDER, (
            "高速版の抽出順が utils.extract_keypoints と食い違っている"
        )


# ランドマーク ID ごとの位置。解剖学的にもっともらしい配置。単位は m。
# ここから任意の pose_keypoints 構成の点列を組み立てられる。
LANDMARK_POSITIONS = {
    11: [-0.15, 0.0, 1.40],   # 左肩
    12: [0.15, 0.0, 1.40],    # 右肩
    13: [-0.18, 0.0, 1.16],   # 左肘（上腕 0.24）
    14: [0.18, 0.0, 1.16],    # 右肘
    15: [-0.20, 0.0, 0.95],   # 左手首（前腕 0.21）
    16: [0.20, 0.0, 0.95],    # 右手首
    17: [-0.24, 0.0, 0.88],   # 左小指 MCP（手 0.09）
    18: [0.24, 0.0, 0.88],    # 右小指 MCP
    19: [-0.20, 0.04, 0.87],  # 左人差指 MCP
    20: [0.20, 0.04, 0.87],   # 右人差指 MCP
    21: [-0.18, 0.05, 0.92],  # 左親指
    22: [0.18, 0.05, 0.92],   # 右親指
    23: [-0.10, 0.0, 1.00],   # 左腰
    24: [0.10, 0.0, 1.00],    # 右腰
    25: [-0.10, 0.0, 0.60],   # 左膝（大腿 0.40）
    26: [0.10, 0.0, 0.60],    # 右膝
    27: [-0.10, 0.0, 0.20],   # 左足首
    28: [0.10, 0.0, 0.20],    # 右足首
}


def _skeleton(keypoints=None) -> np.ndarray:
    """指定した構成の点列を、抽出と同じ ID 昇順で組み立てる。"""
    ids = sorted(pose_keypoints if keypoints is None else keypoints)
    return np.array([LANDMARK_POSITIONS[pid] for pid in ids], dtype=float)


def _synthetic_skeleton() -> np.ndarray:
    """現行構成（`config.pose_keypoints`）の点列。"""
    return _skeleton()


# 名前ごとの、もっともらしいリンク長の範囲 [m]
EXPECTED_SPAN = {
    "upper_arm_R": (0.15, 0.40), "forearm_R": (0.13, 0.35),
    "up_arm_l": (0.15, 0.40), "forearm_L": (0.13, 0.35),
    "both_shoulder": (0.20, 0.50), "both_hip": (0.12, 0.40),
    "upper_Leg_R": (0.30, 0.55), "upper_Leg_L": (0.30, 0.55),
}


def _config_links() -> dict[str, tuple[int, int]]:
    from config import part_calculations

    return {k: (v["start"], v["end"]) for k, v in part_calculations.items()}


def _phone_links() -> dict[str, tuple[int, int]]:
    from app.runners.network_measure import PART_LINKS

    return dict(PART_LINKS)


class TestPartLinksAreAnatomical:
    """リンク定義が解剖学的に妥当な 2 点を結ぶ。"""

    @pytest.mark.parametrize("loader", [_config_links, _phone_links],
                             ids=["config", "network_measure"])
    def test_every_link_has_a_plausible_length(self, loader):
        points = _synthetic_skeleton()
        for name, (start, end) in loader().items():
            span = float(np.linalg.norm(points[end] - points[start]))
            low, high = EXPECTED_SPAN[name]
            assert low <= span <= high, (
                f"{name} が {_slot_name(start)}→{_slot_name(end)} を結んでおり、"
                f" 長さ {span:.3f} m が想定 {low}〜{high} m から外れる。"
                " 体を斜めに横切るリンクになっていないか確認すること"
            )

    def test_the_phone_path_derives_from_config(self):
        """スマホ経路のリンク定義が config から派生している。"""
        assert _config_links() == _phone_links(), (
            "config.part_calculations と PART_LINKS が食い違っている"
        )

    def test_the_main_script_has_no_rival_definition(self):
        """``master_research_code.py`` が独自の part_calculations を持たない。

        かつては同じ辞書が master_research_code.py・master_research_code_00.py・
        network_measure.py に 3 つあり、片方だけ直すと食い違う状態だった。
        正本は config.py 一つに集約してある（再検算 H-9）。
        """
        tree = ast.parse(io.open(MAIN_SCRIPT, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                assert not any(getattr(t, "id", None) == "part_calculations" for t in node.targets), (
                    "master_research_code.py に part_calculations の定義が復活している。"
                    " 正本は config.py"
                )

    def test_every_link_carries_a_centre_of_mass_fraction(self):
        """全リンクが重心比を持ち、値が妥当な範囲にある。"""
        from config import part_calculations

        for name, spec in part_calculations.items():
            assert "com_fraction" in spec, f"{name} に com_fraction が無い"
            frac = spec["com_fraction"]
            assert 0.0 < frac < 1.0, f"{name} の com_fraction が範囲外: {frac}"

    def test_limb_segments_use_anatomical_fractions(self):
        """四肢は中点（0.5）ではなく文献値を使う。"""
        from config import COM_FRACTIONS, part_calculations

        for name in ("upper_arm_R", "up_arm_l"):
            assert part_calculations[name]["com_fraction"] == COM_FRACTIONS["upper_arm"]
        for name in ("forearm_R", "forearm_L"):
            assert part_calculations[name]["com_fraction"] == COM_FRACTIONS["forearm"]
        # 両肩・両腰は体節ではなく中点なので 0.5 のまま
        for name in ("both_shoulder", "both_hip"):
            assert part_calculations[name]["com_fraction"] == 0.5, (
                f"{name} は両端の中点であるべき（r_g の組み立て側が 3:1 で重み付けする）"
            )


class TestTorqueLinksHandedness:
    """局所トルクの基準リンクが左右を取り違えていない。"""

    def test_right_side_links_use_right_side_joints(self):
        from app.runners.network_measure import arm_axes

        points = _synthetic_skeleton()
        for side in ("R", "L"):
            for joint, (link, _) in arm_axes(points, side).items():
                span = float(np.linalg.norm(link))
                assert 0.13 <= span <= 0.40, (
                    f"{joint}_{side} の長さ {span:.3f} m が上肢のリンクとして不自然"
                )

    def test_right_and_left_are_mirror_images(self):
        """左右対称な骨格なら、対応するリンクと親は x 成分だけが反転する。"""
        from app.runners.network_measure import arm_axes

        points = _synthetic_skeleton()
        right, left = arm_axes(points, "R"), arm_axes(points, "L")
        for joint in ("wrist", "elbow", "shoulder"):
            for which, vr, vl in (("link", right[joint][0], left[joint][0]),
                                  ("parent", right[joint][1], left[joint][1])):
                assert np.allclose(vr, vl * np.array([-1.0, 1.0, 1.0])), (
                    f"{joint} の {which} が左右で鏡像になっていない: {vr} / {vl}。"
                    " 左右の索引を取り違えていないか確認すること"
                )


class TestLinkDefinitionsSurviveAddedLandmarks:
    """リンク定義がランドマーク ID で書かれており、点を足しても壊れない。

    `config.PART_LINK_IDS` が正本で、`part_calculations` はそこから
    `build_part_calculations()` で組み立てられる。位置索引を直書きすると
    `pose_keypoints` に点を足したときに別の関節を指すようになる（再検算 R-1 と同じ壊れ方）。
    """

    # 手のランドマークを足した構成。A-3 でこれを既定にする予定
    WITH_HANDS = sorted(list(pose_keypoints) + [17, 18, 19, 20])

    def test_part_calculations_are_derived_not_literal(self):
        """`part_calculations` が `PART_LINK_IDS` から導出されている。"""
        from config import MP_LANDMARK, SLOT, PART_LINK_IDS, part_calculations

        assert set(part_calculations) == set(PART_LINK_IDS), "リンク名の集合が食い違う"
        for name, (start, end, com) in PART_LINK_IDS.items():
            spec = part_calculations[name]
            assert spec["start"] == SLOT[MP_LANDMARK[start]], f"{name} の start が導出値と違う"
            assert spec["end"] == SLOT[MP_LANDMARK[end]], f"{name} の end が導出値と違う"
            assert spec["com_fraction"] == com, f"{name} の com_fraction が導出値と違う"

    def test_links_connect_the_same_joints_after_adding_hands(self):
        """手のランドマークを足しても、各リンクは同じ関節どうしを結ぶ。"""
        from config import MP_LANDMARK, PART_LINK_IDS, build_part_calculations

        extended = build_part_calculations(self.WITH_HANDS)
        order = sorted(self.WITH_HANDS)
        for name, (start, end, _) in PART_LINK_IDS.items():
            spec = extended[name]
            assert order[spec["start"]] == MP_LANDMARK[start], (
                f"{name} の start が {LANDMARK_NAME[order[spec['start']]]} を指している"
                f"（期待 {LANDMARK_NAME[MP_LANDMARK[start]]}）"
            )
            assert order[spec["end"]] == MP_LANDMARK[end], (
                f"{name} の end が {LANDMARK_NAME[order[spec['end']]]} を指している"
                f"（期待 {LANDMARK_NAME[MP_LANDMARK[end]]}）"
            )

    def test_links_stay_anatomical_after_adding_hands(self):
        """手を足した構成でも、リンク長が解剖学的に妥当なまま。"""
        from config import build_part_calculations

        points = _skeleton(self.WITH_HANDS)
        for name, spec in build_part_calculations(self.WITH_HANDS).items():
            span = float(np.linalg.norm(points[spec["end"]] - points[spec["start"]]))
            low, high = EXPECTED_SPAN[name]
            assert low <= span <= high, (
                f"{name} が {_slot_name(spec['start'], self.WITH_HANDS)}"
                f"→{_slot_name(spec['end'], self.WITH_HANDS)} を結んでおり、"
                f" 長さ {span:.3f} m が想定 {low}〜{high} m から外れる"
            )

    def test_hard_coded_indices_would_break(self):
        """対照: 位置索引を直書きしていたら手を足した時点で壊れる。

        このテストが落ちるようになったら、それは索引がずれない構成に
        変わったということなので、上の 3 つと合わせて見直すこと。
        """
        literal = _config_links()          # 現行構成での位置索引
        order = sorted(self.WITH_HANDS)    # 手を足したときの並び
        hands = {17, 18, 19, 20}
        pointing_at_hands = [
            name for name, (start, end) in literal.items()
            if order[start] in hands or order[end] in hands
        ]
        assert pointing_at_hands, (
            "手を足しても直書き索引が 1 つも手のランドマークを指さない。"
            " 前提（腰・膝・足首がずれる）が崩れている"
        )
        # 実際にずれるのは下肢と胴体。上肢は [0..5] のままなので無傷
        assert "both_hip" in pointing_at_hands, (
            f"both_hip が手を指していない（実際に壊れたのは {pointing_at_hands}）"
        )

    def test_missing_landmark_fails_loudly(self):
        """`pose_keypoints` に無い点を指したら KeyError で落ちる。"""
        from config import build_part_calculations

        # 手首を外した構成では forearm_* が引けない
        without_wrist = [p for p in pose_keypoints if p not in (15, 16)]
        with pytest.raises(KeyError):
            build_part_calculations(without_wrist)
