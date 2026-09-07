"""設定システムの契約を検証する。

``master_research_code.py`` には ``os.getenv`` が 150 箇所（ユニーク 135 個）あり、
アプリはそれを**環境変数として子プロセスに渡す**ことで、既存コードを 1 行も
変えずに制御する。ここが枠組みの中心なので、契約を厳しく固定する。
"""

from __future__ import annotations

import json

import pytest

from app.core import settings as st

# コードの既定値が壊れていて、アプリ側で上書きが必要なフラグ。
# 素の `python master_research_code.py` はこれらが有効なせいで、
# 逆動力学が無効・同一画像で三角測量・周波数不整合・フィルタ差し替えが起きる。
BROKEN_DEFAULT_FLAGS = [
    "DEMO_MONO_GAUGE_ON",
    "DEMO_MONO_CAM0_ONLY",
    "RT_POSE_FIXED_HZ_ON",
    "E_LPF_NATIVE_ON",
]


class TestSchema:
    def test_schema_is_not_empty(self):
        assert len(st.SCHEMA) > 100, "スキーマの読み込みに失敗している"

    def test_every_setting_has_known_type(self):
        for name, setting in st.SCHEMA.items():
            assert setting.type in ("bool", "int", "float", "str"), f"{name} の型が不正"

    @pytest.mark.parametrize("name", BROKEN_DEFAULT_FLAGS)
    def test_broken_flags_are_overridden(self, name):
        """コード側の既定が '1'、アプリ側の既定が '0' になっていること。"""
        setting = st.SCHEMA[name]
        assert setting.code_default == "1", f"{name} のコード既定が想定と違う"
        assert setting.app_default == "0", f"{name} をアプリ側で無効化していない"
        assert setting.effective_default == "0"

    @pytest.mark.parametrize("name", BROKEN_DEFAULT_FLAGS)
    def test_broken_flags_are_visible_and_explained(self, name):
        """利用者が「なぜ既定を変えたか」を UI で読めること。黙って変えない。"""
        setting = st.SCHEMA[name]
        assert setting.ui_visible, f"{name} が UI に出ない"
        assert setting.description, f"{name} に説明が無い"


class TestTypedAccess:
    def test_bool_reads_as_python_bool(self):
        s = st.Settings()
        assert s.get("DEMO_MONO_GAUGE_ON") is False

    def test_set_and_get_bool(self):
        s = st.Settings()
        s.set("DEMO_MONO_GAUGE_ON", True)
        assert s.get("DEMO_MONO_GAUGE_ON") is True

    def test_int_and_float_are_coerced(self):
        s = st.Settings()
        for name, setting in st.SCHEMA.items():
            if setting.type == "int" and setting.effective_default is not None:
                assert isinstance(s.get(name), int)
                break
        for name, setting in st.SCHEMA.items():
            if setting.type == "float" and setting.effective_default is not None:
                assert isinstance(s.get(name), float)
                break

    def test_unknown_name_raises_with_helpful_message(self):
        s = st.Settings()
        with pytest.raises(KeyError) as exc:
            s.get("NO_SUCH_SETTING")
        assert "NO_SUCH_SETTING" in str(exc.value)

    def test_set_rejects_wrong_type(self):
        s = st.Settings()
        with pytest.raises((TypeError, ValueError)):
            s.set("DEMO_MONO_GAUGE_ON", "たぶん")


class TestEnvExport:
    def test_all_values_are_strings(self):
        env = st.Settings().as_env()
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())

    def test_exports_every_setting_not_just_changed_ones(self):
        """**全件**を明示的に渡すこと。

        差分だけ渡すと、渡さなかった項目は子プロセス側の既定値
        （＝壊れている方）が効いてしまう。子プロセスの挙動を完全に決めるため、
        呼び出し元のシェル環境に依存させない。
        """
        env = st.Settings().as_env()
        for name, setting in st.SCHEMA.items():
            if setting.effective_default is not None:
                assert name in env, f"{name} が環境変数に含まれていない"

    @pytest.mark.parametrize("name", BROKEN_DEFAULT_FLAGS)
    def test_exported_bool_is_falsey_for_original_idiom(self, name):
        """既存コードの ``os.getenv(X, '1') in ('1','true','True')`` で False になること。

        アプリ側が '0' を渡しても、既存コードの読み方で True になってしまっては
        意味がない。実際の読み取り方をそのまま再現して確かめる。
        """
        env = st.Settings().as_env()
        assert env[name] not in ("1", "true", "True"), f"{name} が有効のまま"

    def test_changed_values_are_exported(self):
        s = st.Settings()
        s.set("DEMO_MONO_GAUGE_ON", True)
        assert s.as_env()["DEMO_MONO_GAUGE_ON"] in ("1", "true", "True")


class TestPersistence:
    def test_round_trip_preserves_changes(self, tmp_path):
        path = tmp_path / "settings.json"
        original = st.Settings()
        original.set("DEMO_MONO_GAUGE_ON", True)
        original.save(path)

        loaded = st.Settings.load(path)
        assert loaded.get("DEMO_MONO_GAUGE_ON") is True

    def test_saves_only_differences_from_default(self, tmp_path):
        """既定と同じ値まで書き出すと、後で既定を変えたときに追随できなくなる。"""
        path = tmp_path / "settings.json"
        s = st.Settings()
        s.set("DEMO_MONO_GAUGE_ON", True)
        s.save(path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert list(payload["values"]) == ["DEMO_MONO_GAUGE_ON"]

    def test_load_missing_file_returns_defaults(self, tmp_path):
        s = st.Settings.load(tmp_path / "does_not_exist.json")
        assert s.get("DEMO_MONO_GAUGE_ON") is False

    def test_load_ignores_settings_that_no_longer_exist(self, tmp_path):
        """コード側から消えた設定が保存ファイルに残っていても起動できること。"""
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"values": {"REMOVED_LONG_AGO": "1"}}), encoding="utf-8"
        )
        s = st.Settings.load(path)
        assert "REMOVED_LONG_AGO" not in s.as_env()
