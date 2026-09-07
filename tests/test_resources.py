"""同梱リソースの解決を検証する。

凍結（PyInstaller）すると、コードは一時展開ディレクトリ ``sys._MEIPASS`` から動く。
一方 ``config.py:7`` の ``folder_path = os.path.dirname(os.path.abspath(__file__))`` は
リポジトリ構成を前提にしており、そのままでは凍結後に資産を見つけられない。
この差を resources.py が吸収する。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from app.core import resources


class TestFrozenDetection:
    def test_not_frozen_during_tests(self):
        assert resources.is_frozen() is False

    def test_detects_frozen_via_meipass(self, monkeypatch, tmp_path):
        """PyInstaller 実行時の経路を、凍結せずに検証する。"""
        monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
        monkeypatch.setattr(resources.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert resources.is_frozen() is True
        assert resources.resource_root() == tmp_path


class TestResourceRoot:
    def test_is_existing_directory(self):
        assert resources.resource_root().is_dir()

    def test_dev_root_is_repo_root(self):
        """開発時はリポジトリルート。目印として既知のファイルの存在を見る。"""
        assert (resources.resource_root() / "requirements_min.txt").is_file()


class TestAssetPath:
    def test_asset_path_is_under_resource_root(self):
        p = resources.asset_path("fonts", "ipaexg.ttf")
        assert resources.resource_root() in p.parents

    def test_require_asset_returns_existing_file(self):
        p = resources.require_asset("fonts", "ipaexg.ttf")
        assert p.is_file()

    def test_require_asset_raises_with_helpful_message(self):
        """欠損時は「何が無いか」が分かるメッセージで落ちること。

        このリポジトリは .gitignore の書き方の問題で必須資産が追跡されておらず、
        clone しても動かない状態だった。無言で握りつぶすと同じ事故が再発する。
        """
        with pytest.raises(FileNotFoundError) as exc:
            resources.require_asset("fonts", "does_not_exist.ttf")
        assert "does_not_exist.ttf" in str(exc.value)


class TestJapaneseFont:
    def test_font_file_is_bundled(self):
        assert resources.japanese_font_path().is_file()

    def test_font_loads_with_pillow(self):
        """PIL で実際に開けること。utils.put_text_jp が同じ経路で使う。"""
        from PIL import ImageFont

        font = ImageFont.truetype(str(resources.japanese_font_path()), 24)
        assert font.getbbox("車椅子")[2] > 0, "日本語の描画幅が 0（グリフが無い）"

    def test_license_is_bundled_alongside(self):
        """IPA フォントライセンスは再配布時の同梱が条件。"""
        license_file = resources.japanese_font_path().parent / "IPA_Font_License_Agreement_v1.0.txt"
        assert license_file.is_file(), "フォントを配布するならライセンス文書も同梱が必要"


class TestMatplotlibJapanese:
    def test_configure_sets_bundled_font(self):
        import matplotlib

        resources.configure_matplotlib_japanese()
        assert matplotlib.rcParams["font.family"], "font.family が設定されていない"

    def test_japanese_text_renders_without_missing_glyph(self):
        """japanize_matplotlib の代替として実際に機能しているかを見る。

        グリフが無いと matplotlib は 'Glyph ... missing from font(s)' を警告する。
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        resources.configure_matplotlib_japanese()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            ax.set_title("右手首トルク")
            ax.set_xlabel("時間（秒）")
            fig.canvas.draw()
            plt.close(fig)

        missing = [w for w in caught if "missing from font" in str(w.message)]
        assert not missing, f"日本語グリフが欠落: {[str(w.message) for w in missing]}"
