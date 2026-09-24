"""ゲージの書体（``app.gauge.fonts``）の試験。

名前の照合と太さの選び方は Qt に依存しないので、そのまま確かめる。Qt 側は、同梱の IPAex ゴシックの
名前を「FOT-ロダン Pro EB」などに書き換えた偽の書体を登録し、Mac にフォントワークスの書体が
入っているのと同じ状況を作って確かめる（この試験の環境には本物が無い）。書体の組は ``FontSet`` ごとに
持つので、試験ごとに新しい ``FontSet`` を作れば、ほかの試験の組に影響しない。
"""

from __future__ import annotations

import pytest

from app.gauge import fonts as gf
from app.gauge.fonts import Face


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
    assert gf.FontSet("rodan").name == gf.DEFAULT_PRESET
    assert "rodan" in capsys.readouterr().err
    assert gf.FontSet("").name == gf.DEFAULT_PRESET
    assert gf.FontSet(None).name == gf.DEFAULT_PRESET
    assert gf.FontSet(" Kaimin ").name == "kaimin"


def test_font_set_is_shared_per_preset():
    # 窓の動かない層は FontSet の同一性で作り直すかを決めるので、同じ組は同じものを返す
    assert gf.font_set(" Kaimin ") is gf.font_set("kaimin")
    assert gf.font_set(None) is gf.font_set(gf.DEFAULT_PRESET)
    assert gf.font_set("system") is not gf.font_set("tsukushi")


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
    _forget_installed_faces()  # 登録した書体を引き直させる
    yield families
    _forget_installed_faces()


def _forget_installed_faces():
    """インストール済みの書体の一覧と、それを引いた使い回しの FontSet を捨てる。"""
    gf._installed_faces.cache_clear()
    gf._shared_font_set.cache_clear()


def test_installed_fontworks_faces_are_used(fake_fontworks):
    fonts = gf.FontSet("rodin")
    heading = fonts.font("heading", 22, 800)
    numeral = fonts.font("numeral", 22, 800)
    text = fonts.font("text", 15, 700)
    assert heading.families()[0] == fake_fontworks["rodin"]
    assert numeral.families()[0] == fake_fontworks["udlarge"]
    assert text.families()[0] == fake_fontworks["udlarge"]
    # 書体そのものの太さを指定する（Qt に太字を合成させない）
    assert heading.weight() == 800
    assert text.weight() == 700
    # 見つからない字（✓ など）のための代わりが後ろに続く
    assert "Hiragino Sans" in heading.families()


def test_system_preset_keeps_the_old_fonts(fake_fontworks):
    font = gf.FontSet("system").font("heading", 22, 800)
    assert font.families()[0] == "Hiragino Sans"
    assert font.weight() == 800


def test_missing_heading_falls_back_to_mincho_then_gothic(fake_fontworks):
    fonts = gf.FontSet("tsukushi")  # 筑紫A見出ミンは登録していない
    font = fonts.font("heading", 22, 800)
    assert font.families()[:2] == ["Hiragino Mincho ProN", "ヒラギノ明朝 ProN"]
    assert fonts.font("numeral", 22, 800).families()[0] == fake_fontworks["udlarge"]


def test_report_lists_the_found_faces(fake_fontworks):
    report = gf._report("rodin")
    assert fake_fontworks["rodin"] in report
    assert "rodin: 見出し ロダン／数字・文字 UD角ゴ_ラージ（選んだ組）" in report
    # 見つからない役は、代わりの並びを出す（筑紫A見出ミンは登録していない）
    assert "見つからない → Hiragino Mincho ProN → ヒラギノ明朝 ProN → Hiragino Sans" in report


def test_window_takes_the_preset_and_begin_changes_it(fake_fontworks):
    from app.gauge.window import GaugeWindow

    window = GaugeWindow(font_preset="tsukushi")
    try:
        assert window.gauge.font_set is gf.font_set("tsukushi")
        window.begin(show_joules=True)  # 渡さなければ今の組のまま
        assert window.gauge.font_set.name == "tsukushi"

        window.begin(show_joules=True, font_preset="rodin")
        fonts = window.gauge.font_set
        assert fonts.name == "rodin"
        # 見出し帯の題（header_title）は見出しの書体で描く
        title = fonts.font(gf.role_for_label("header_title"), 22, 800)
        assert title.families()[0] == fake_fontworks["rodin"]
    finally:
        window.close()
