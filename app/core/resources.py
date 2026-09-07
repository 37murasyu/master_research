"""同梱リソース（フォント・画像・モデル）の在り処を解決する。

開発時はリポジトリルート、PyInstaller で凍結したときは一時展開先 ``sys._MEIPASS``
を基点にする。呼び出し側はどちらで動いているかを意識しない。

**読み取り専用の同梱資産**をここで扱う。**書き込み先**（設定・計測結果）は
``platform_compat.user_config_dir`` / ``user_output_dir`` の担当で、混ぜないこと。
凍結アプリでは実行ファイルの隣に書き込めないため、この区別が要る。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

__all__ = [
    "is_frozen",
    "resource_root",
    "asset_path",
    "require_asset",
    "japanese_font_path",
    "configure_matplotlib_japanese",
]

# 同梱資産を置くディレクトリ名。PyInstaller の spec でもこの名前で追加する。
ASSETS_DIRNAME = "assets"


def is_frozen() -> bool:
    """PyInstaller で固められた状態かどうか。"""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """同梱資産を探す基点。

    凍結時は ``sys._MEIPASS``（実行のたびに変わる一時ディレクトリ）。
    開発時はこのファイルから 2 つ上、すなわちリポジトリルート
    （``app/core/resources.py`` → ``app/core`` → ``app`` → ルート）。
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined] # pylint: disable=protected-access
    return Path(__file__).resolve().parent.parent.parent


def asset_path(*parts: str) -> Path:
    """``assets/`` 配下のパスを組み立てる。存在確認はしない。"""
    return resource_root() / ASSETS_DIRNAME / Path(*parts)


def require_asset(*parts: str) -> Path:
    """``assets/`` 配下のファイルを返す。無ければ理由の分かる例外を投げる。

    黙って握りつぶさないのが要点。このリポジトリは ``.gitignore`` が
    ``media/`` をディレクトリごと除外した後に ``!media/wheelchair_user.png`` で
    戻そうとしていたが、git は親が除外されたファイルを再包含できないため、
    **clone しても画像が存在しない**状態になっていた。無言で失敗すると同じ事故が再発する。
    """
    path = asset_path(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"同梱資産が見つかりません: {path}\n"
            f"  基点: {resource_root()}（{'凍結' if is_frozen() else '開発'}モード）\n"
            f"  凍結ビルドなら packaging/app.spec の datas に追加されているか確認してください。"
        )
    return path


@lru_cache(maxsize=1)
def japanese_font_path() -> Path:
    """日本語描画に使うフォントファイル。

    IPAexゴシックを同梱している。理由:

    - **Meiryo は同梱できない**。Microsoft の商用フォントで、アプリに埋め込んで
      配布するとライセンス違反になる。旧コードは ``utils.py:217`` などで
      ``folder_path + "\\\\meiryo\\\\meiryo.ttc"`` を参照していた
    - IPA フォントライセンス v1.0 は再配布を明示的に許可している
      （ライセンス文書を同梱すること。``assets/fonts/`` に置いてある）
    - ``japanize-matplotlib`` への依存も外せる。あれは ``distutils.version`` を
      import するため **Python 3.12 では動かない**（distutils が標準ライブラリから削除された）
    """
    return require_asset("fonts", "ipaexg.ttf")


def configure_matplotlib_japanese() -> None:
    """matplotlib が日本語を豆腐にしないよう、同梱フォントを登録する。

    ``import japanize_matplotlib`` の置き換え。冪等なので何度呼んでもよい。
    """
    import matplotlib
    from matplotlib import font_manager

    font_path = japanese_font_path()
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()

    matplotlib.rcParams["font.family"] = family
    # マイナス記号を ASCII のハイフンにする。日本語フォントには U+2212 が
    # 無いことが多く、負のトルク値の軸ラベルが豆腐になるのを防ぐ。
    matplotlib.rcParams["axes.unicode_minus"] = False
