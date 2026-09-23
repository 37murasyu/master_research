"""ゲージ向けの設定 2 件（J の数値の表示・体重）を検証する。

``GAUGE_SHOW_JOULES`` はスキーマ側に無い新しい設定で、CURATED だけで型・既定・
グループ・説明を与える（スイッチのみが操作口で、設定フォームには出さない）。
``BODY_MASS_KG`` はスキーマに既にある項目を、UI に出して説明を上書きするだけ
（既定値の 65 は変えない）。
"""

from __future__ import annotations

from app.core import settings as st


class TestGaugeShowJoules:
    def test_gauge_show_joules_is_bool_default_on(self):
        setting = st.SCHEMA["GAUGE_SHOW_JOULES"]
        assert setting.type == "bool"
        assert setting.effective_default == "1"
        assert st.Settings().get("GAUGE_SHOW_JOULES") is True

    def test_gauge_show_joules_is_not_in_the_form(self):
        assert st.SCHEMA["GAUGE_SHOW_JOULES"].ui_visible is False

    def test_gauge_show_joules_survives_save_and_load(self, tmp_path):
        path = tmp_path / "settings.json"
        original = st.Settings()
        original.set("GAUGE_SHOW_JOULES", False)
        original.save(path)

        loaded = st.Settings.load(path)
        assert loaded.get("GAUGE_SHOW_JOULES") is False


class TestBodyMassKg:
    def test_body_mass_is_visible_with_default_65(self):
        setting = st.SCHEMA["BODY_MASS_KG"]
        assert setting.ui_visible is True
        assert setting.group == "被験者"
        assert setting.effective_default == "65"
        assert st.Settings().get("BODY_MASS_KG") == 65.0
