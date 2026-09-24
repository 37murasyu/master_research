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

    @pytest.mark.parametrize("name, value", [
        ("BODY_MASS_KG", 65.0),  # 既定は "65"。欄は 65.0 を渡す
        ("EKF_Q_ACC", 0.001),  # 既定は "1e-3"
        ("SKIP_FRAMES", 0),
        ("HYBRID_REPLAY_SPEED", 1.0),
    ])
    def test_setting_the_default_as_a_value_keeps_no_difference(self, name, value, tmp_path):
        """既定値は文字列でなく値で比べる。"65.0" と "65" を別の値として毎回差分に保存していた。"""
        s = st.Settings()
        s.set(name, value)
        assert name not in s.overrides

        path = tmp_path / "settings.json"
        s.save(path)
        assert json.loads(path.read_text(encoding="utf-8"))["values"] == {}

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


class TestLoadingBrokenFiles:
    """手で書き換えた・壊れた設定ファイルでも、GUI は既定値で立ち上がり、開始もできること。

    かつては UTF-8 でない・一番外が配列・"65kg"・数や真偽の JSON 値で、起動時か開始時に例外で止まっていた。
    読めないものは項目ごとに捨てて既定値に戻す。
    """

    def _load(self, tmp_path, content) -> st.Settings:
        path = tmp_path / "settings.json"
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        return st.Settings.load(path)

    @pytest.mark.parametrize("content", [
        '{"values": {"SUBJECT_ID": "被験者3"}}'.encode("cp932"),  # UTF-8 でない
        b"\xff\xfe\x00",
        b"[]",
        b'"values"',
        b"3",
        b'{"values": ["SUBJECT_ID"]}',
        b'{"values": "SUBJECT_ID"}',
        b'{"values": null}',
        b"[" * 100_000,  # 深すぎる入れ子
    ], ids=["cp932", "壊れたバイト列", "配列", "文字列", "数", "values が配列", "values が文字列",
            "values が null", "深い入れ子"])
    def test_unreadable_files_give_the_defaults(self, tmp_path, content):
        s = self._load(tmp_path, content)
        assert s.overrides == {}

    @pytest.mark.parametrize("name, value", [
        ("BODY_MASS_KG", "65kg"),
        ("BODY_MASS_KG", "nan"),
        ("BODY_MASS_KG", True),
        ("BODY_MASS_KG", [70]),
        ("MP_THREADS", "auto"),
        ("MP_THREADS", 2.5),
        ("SKIP_FRAMES", "1.5"),
        ("HEADLESS", "たぶん"),
        ("HEADLESS", {"on": True}),
        ("SUBJECT_ID", None),
        ("SUBJECT_ID", False),
    ])
    def test_an_unreadable_value_falls_back_to_the_default_only_for_that_item(self, tmp_path, name, value):
        s = self._load(tmp_path, {"values": {name: value, "CAM0": "1"}})
        assert name not in s.overrides
        assert s.get(name) == st.Settings().get(name)
        assert s.get("CAM0") == "1", "読める項目まで捨てている"

    def test_numbers_and_booleans_in_the_file_are_read_as_their_values(self, tmp_path):
        s = self._load(tmp_path, {"values": {
            "HEADLESS": True, "SUBJECT_ID": 7, "BODY_MASS_KG": 70, "SKIP_FRAMES": 2, "EKF_Q_ACC": "2e-3",
        }})
        assert s.get("HEADLESS") is True
        assert s.get("SUBJECT_ID") == "7"
        assert s.get("BODY_MASS_KG") == 70.0
        assert s.get("SKIP_FRAMES") == 2
        assert s.get("EKF_Q_ACC") == pytest.approx(2e-3)
        env = s.as_env()
        assert all(isinstance(value, str) for value in env.values()), "子への環境変数に文字列でない値がある"
        assert (env["HEADLESS"], env["SUBJECT_ID"], env["SKIP_FRAMES"]) == ("1", "7", "2")
        assert float(env["BODY_MASS_KG"]) == 70.0

    @pytest.mark.parametrize("raw", ["1", "0", "true", "True", "TRUE", "false", "FALSE", "yes", "no", "on", "Off",
                                     " 1 ", ""])
    def test_booleans_are_read_the_same_way_as_the_child(self, tmp_path, raw):
        """GUI と子（config.env_flag）で真偽の読み方をそろえる。"TRUE" で GUI は偽・子は真と食い違っていた。"""
        from config import env_flag

        for name in ("HEADLESS", "DEMO_MONO_GAUGE_ON", "GAUGE_SHOW_JOULES"):
            default = st.Settings().get(name)
            s = self._load(tmp_path, {"values": {name: raw}})
            child = env_flag(name, default, env=s.as_env())
            assert s.get(name) is child, (name, raw)
            assert s.get(name) is env_flag(name, default, env={name: raw}), (name, raw)


class TestAtomicSave:
    def test_a_failed_save_keeps_the_previous_file(self, tmp_path, monkeypatch):
        """書き込みの途中で止まっても、前回の設定ファイルは壊れない（一時ファイルに書いてから置き換える）。"""
        import os

        path = tmp_path / "settings.json"
        before = st.Settings()
        before.set("SUBJECT_ID", "3")
        before.save(path)
        saved = path.read_bytes()

        def fail(*_args, **_kwargs):
            raise OSError("ディスクがいっぱい")

        monkeypatch.setattr(os, "replace", fail)
        after = st.Settings()
        after.set("SUBJECT_ID", "4")
        with pytest.raises(OSError):
            after.save(path)

        assert path.read_bytes() == saved
        assert [p.name for p in tmp_path.iterdir()] == ["settings.json"], "一時ファイルが残った"

    def test_save_replaces_the_file_from_the_same_folder(self, tmp_path, monkeypatch):
        import os

        replaced = []
        real_replace = os.replace
        monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((src, dst)) or real_replace(src, dst))
        path = tmp_path / "sub" / "settings.json"
        st.Settings().save(path)

        assert len(replaced) == 1
        src, dst = map(os.fspath, replaced[0])
        assert os.path.dirname(src) == os.path.dirname(dst) == str(path.parent)
        assert st.Settings.load(path).overrides == {}
