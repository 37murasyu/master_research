"""混成（Mac＋Pixel）だけが読む設定が、GUI から子プロセスへ既定値つきで渡ることを確かめる。

**なぜこのテストがあるか。**

GUI は ``Settings.as_env()`` の全件を子プロセスへ渡す（``app.entry.worker_environment``）。ところが設定スキーマの
生成元（``tools/extract_env_schema.py``）は ``master_research_code.py`` と ``config.py`` しか見ないので、
混成の経路だけが読む環境変数はスキーマに入らず、``settings.CURATED`` に足さない限り GUI から渡らない。

名前を ``HYBRID_`` で始めるのは、GUI が USB 経路向けの既定（``POSE_ROI_ON=1``・``MP_INPUT_SCALE=0.5``・
``EKF_Q_ACC=1e-3`` など）を全件渡すため。混成が USB と同じ名前を読むと、その既定が黙って混成に効いてしまう
（2026-09-24 の計画。合成の押し上げで EKF の 1e-3 は肘の仕事を 52% 過大にした）。
"""

from __future__ import annotations

import pytest

from app.core.settings import SCHEMA, Settings

# 名前 → (型, 既定値, 画面に出すか)
HYBRID_SETTINGS = {
    "HYBRID_EKF_PROFILE": ("str", "", True),
    "HYBRID_GRAVITY_BOARD": ("bool", "1", True),
    "HYBRID_GRAVITY_BOARD_TIMEOUT_S": ("float", "30", False),
    "HYBRID_POSE_MODEL": ("str", "", False),
    "HYBRID_POSE_MIN_DET": ("float", "0.5", False),
    "HYBRID_POSE_MIN_PRESENCE": ("float", "0.5", False),
    "HYBRID_POSE_MIN_TRACK": ("float", "0.5", False),
    "HYBRID_POSE_INPUT_SCALE": ("float", "1.0", False),
    "HYBRID_POSE_ROI": ("bool", "0", False),
    "HYBRID_DYN_GATE": ("bool", "1", False),
}


@pytest.mark.parametrize("name", sorted(HYBRID_SETTINGS))
def test_hybrid_setting_reaches_the_worker_with_its_default(name):
    kind, default, _ = HYBRID_SETTINGS[name]
    assert name in SCHEMA, f"{name} が設定に無い（CURATED に足していない）ので GUI から子へ渡らない"
    assert SCHEMA[name].type == kind
    assert Settings().as_env()[name] == default


@pytest.mark.parametrize("name", sorted(HYBRID_SETTINGS))
def test_only_the_operator_facing_hybrid_settings_are_shown(name):
    """画面に出すのは EKF のプロファイルと盤を立てる校正だけ（他は現場で触らない調整値）。"""
    assert SCHEMA[name].ui_visible is HYBRID_SETTINGS[name][2]


def test_one_rm_table_path_is_not_forced_on_the_worker():
    """1RM の表の場所は既定では渡さない（子が config.folder_path の m_max_all_merged.csv を探す）。"""
    assert "ONE_RM_CSV" in SCHEMA
    assert "ONE_RM_CSV" not in Settings().as_env()
