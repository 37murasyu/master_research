"""ゲージの書体（``app.gauge.fonts``）の試験。

名前の照合と太さの選び方は Qt に依存しないので、そのまま確かめる。Qt 側は、同梱の IPAex ゴシックの
名前を「FOT-ロダン Pro EB」などに書き換えた偽の書体を登録し、Mac にフォントワークスの書体が
入っているのと同じ状況を作って確かめる（この試験の環境には本物が無い）。
"""

from __future__ import annotations

import pytest

from app.gauge import fonts as gf
from app.gauge.fonts import Face


@pytest.fixture(autouse=True)
def _restore_preset():
    before = gf.current_preset()
    yield
    gf.set_preset(before)


# ---------------------------------------------------------------------------
# 名前の照合（Qt なし）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [
    "FOT-ロダン Pro EB", "FOT-Rodin Pro", "FOT-RodinPro-EB", "ＦＯＴ－ロダン Ｐｒｏ ＥＢ",
])
def test_rodin_is_found_under_its_various_names(family):
    assert gf.matches(gf.RODIN, family)


@pytest.mark.parametrize("family", ["FOT-筑紫A見出ミン Std E", "FOT-TsukuAMidashiMinStd-E"])
def test_tsukushi_midashi_is_found(family):
    # 英語名は Tsukushi とも、略して Tsuku とも書かれる
    assert gf.matches(gf.TSUKUSHI_MIDASHI, family)


@pytest.mark.parametrize("family", ["FOT-UD角ゴ_ラージ Pr6N B", "FOT-UDKakugo_Large Pr6N", "FOT-UD角ゴ ラージ Pr6N"])
def test_ud_kakugo_large_is_found(family):
    assert gf.matches(gf.UD_KAKUGO_LARGE, family)


def test_ud_kakugo_large_is_not_confused_with_plain_ud_kakugo():
    assert not gf.matches(gf.UD_KAKUGO_LARGE, "FOT-UD角ゴC80 Pro B")


@pytest.mark.parametrize("family, ok", [
    ("FOT-ロダン Pro EB", True),
    ("FOT-Rodin Pro", True),
    ("FOT-ニューロダン Pro EB", False),
    ("FOT-NewRodin Pro", False),
    ("FOT-ロダンわんぱく Std", False),
])
def test_rodin_excludes_its_relatives(family, ok):
    assert gf.matches(gf.RODIN, family) is ok


@pytest.mark.parametrize("family, style, weight", [
    ("FOT-ロダン Pro EB", "Regular", 800),  # 太さごとに別の family。スタイル名は飾り
    ("FOT-UD角ゴ_ラージ Pr6N B", "", 700),
    ("FOT-Rodin Pro", "DB", 600),
    ("FOT-筑紫A見出ミン Std E", "", 800),
    ("FOT-Rodin Pro", "UB", 900),
    ("Hiragino Sans", "W6", None),
])
def test_weight_is_read_from_the_name(family, style, weight):
    assert gf._weight_from_name(family, style) == weight


def test_pick_face_takes_the_nearest_weight_and_prefers_heavier_on_ties():
    faces = [Face("FOT-Rodin Pro", "DB", 600), Face("FOT-Rodin Pro", "EB", 800), Face("Other", "", 700)]
    assert gf.pick_face((gf.RODIN,), faces, 800).style == "EB"
    assert gf.pick_face((gf.RODIN,), faces, 700).style == "EB", "同じ近さなら太い方"
    assert gf.pick_face((gf.RODIN,), faces, 400).style == "DB"
    assert gf.pick_face((gf.KAIMIN_SORA,), faces, 800) is None


@pytest.mark.parametrize("label_role, role", [
    ("header_title", "heading"),
    ("value_text", "numeral"), ("header_rep", "numeral"), ("band_label", "numeral"),
    ("part", "text"), ("state", "text"), ("legend_band_label", "text"), ("header_rep_sub", "text"),
])
def test_role_for_label(label_role, role):
    assert gf.role_for_label(label_role) == role


def test_unknown_preset_falls_back_to_default(capsys):
    assert gf.set_preset("rodan") == gf.DEFAULT_PRESET
    assert "rodan" in capsys.readouterr().err
    assert gf.set_preset("") == gf.DEFAULT_PRESET
    assert gf.set_preset(" Kaimin ") == "kaimin"


def test_changing_preset_bumps_generation():
    gf.set_preset("system")
    before = gf.font_generation()
    gf.set_preset("system")
    assert gf.font_generation() == before, "同じ組なら動かない層を作り直さない"
    gf.set_preset("tsukushi")
    assert gf.font_generation() == before + 1


def test_setting_defaults_to_rodin():
    from app.core.settings import Settings

    assert Settings().get("GAUGE_FONT_PRESET") == "rodin"
    assert set(gf.PRESETS) == {"rodin", "tsukushi", "kaimin", "system"}


# ---------------------------------------------------------------------------
# Qt 側（偽のフォントワークスの書体を登録する）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_fontworks(tmp_path_factory):
    """IPAex ゴシックの名前を書き換えた偽の書体を 2 つ登録する（ロダン EB、UD角ゴ_ラージ B）。"""
    pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")
    ttlib = pytest.importorskip("fontTools.ttLib")
    from app.core.qt import QtGui, QtWidgets
    from app.core.resources import japanese_font_path

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    directory = tmp_path_factory.mktemp("fonts")
    families = {}
    for stem, family, family_en, weight in (
        ("rodin", "FOT-ロダン Pro EB", "FOT-Rodin Pro EB", 800),
        ("udlarge", "FOT-UD角ゴ_ラージ Pr6N B", "FOT-UDKakugo_Large Pr6N B", 700),
    ):
        font = ttlib.TTFont(str(japanese_font_path()))
        name = font["name"]
        for record in list(name.names):
            if record.nameID in (1, 4, 16):
                # Mac の名前の表（platformID 1）は日本語を書けないので英語名にする
                text = family if record.platformID == 3 else family_en
                name.setName(text, record.nameID, record.platformID, record.platEncID, record.langID)
            elif record.nameID == 6:
                name.setName(stem, record.nameID, record.platformID, record.platEncID, record.langID)
        font["OS/2"].usWeightClass = weight
        path = directory / f"{stem}.ttf"
        font.save(str(path))
        font_id = QtGui.QFontDatabase.addApplicationFont(str(path))
        assert font_id != -1
        families[stem] = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]
    gf._FACES = None  # 登録した書体を引き直させる
    gf._FONT_CACHE.clear()
    yield families
    gf._FACES = None
    gf._FONT_CACHE.clear()


def test_installed_fontworks_faces_are_used(fake_fontworks):
    gf.set_preset("rodin")
    heading = gf.make_font("heading", 22, 800)
    numeral = gf.make_font("numeral", 22, 800)
    text = gf.make_font("text", 15, 700)
    assert heading.families()[0] == fake_fontworks["rodin"]
    assert numeral.families()[0] == fake_fontworks["udlarge"]
    assert text.families()[0] == fake_fontworks["udlarge"]
    # 書体そのものの太さを指定する（Qt に太字を合成させない）
    assert heading.weight() == 800
    assert text.weight() == 700
    # 見つからない字（✓ など）のための代わりが後ろに続く
    assert "Hiragino Sans" in heading.families()


def test_system_preset_keeps_the_old_fonts(fake_fontworks):
    gf.set_preset("system")
    font = gf.make_font("heading", 22, 800)
    assert font.families()[0] == "Hiragino Sans"
    assert font.weight() == 800


def test_missing_heading_falls_back_to_mincho_then_gothic(fake_fontworks):
    gf.set_preset("tsukushi")  # 筑紫A見出ミンは登録していない
    font = gf.make_font("heading", 22, 800)
    assert font.families()[:2] == ["Hiragino Mincho ProN", "ヒラギノ明朝 ProN"]
    assert gf.make_font("numeral", 22, 800).families()[0] == fake_fontworks["udlarge"]


def test_report_lists_the_found_faces(fake_fontworks):
    gf.set_preset("rodin")
    report = gf._report()
    assert fake_fontworks["rodin"] in report
    assert "（今の設定）" in report


def test_widget_uses_the_heading_font_for_the_title(fake_fontworks):
    from app.gauge import scene as sc
    from app.gauge.widget import _run_font
    from app.shell import theme

    gf.set_preset("rodin")
    font = _run_font(sc.Run("上肢の仕事量", 22, 800, theme.TEXT), "header_title")
    assert font.families()[0] == fake_fontworks["rodin"]
