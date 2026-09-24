"""色の役割とコントラストを検査する ``app.shell.theme`` の試験。

``app/shell/theme.py`` は、この後の scene・widget・計測画面の部品が色を取り出す
唯一の場所になる予定なので、Qt には依存しない（このファイルも依存しない）。

確かめるのは 2 つ。
1. WCAG 2.x の相対輝度・コントラスト比の計算そのものが正しいこと。
2. 実際の色の組が、文字なら 4.5、図形なら 3.0 の基準を満たすこと
   （満たさない既知の組は、理由つきで ``KNOWN_LOW_CONTRAST`` に列挙されていて、
   それを除けば残りはすべて基準を満たすこと）。
"""

from __future__ import annotations

import pytest

from app.shell import theme


def test_contrast_ratio_known_values():
    # 白と黒はどちらの順で渡しても 21（WCAG のコントラスト比の最大値）。
    assert theme.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0, abs=0.01)
    assert theme.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    # 同じ色どうしは 1。
    for color in (theme.FIELD, theme.BAND, theme.AMBER, "#ffffff"):
        assert theme.contrast_ratio(color, color) == pytest.approx(1.0, abs=1e-9)


def test_text_on_field_meets_4_5():
    # 文字・補足・過負荷（の文字表示）は地の上で「文字」の基準を満たす。
    for color in (theme.TEXT, theme.SUBTEXT, theme.OVER):
        ratio = theme.contrast_ratio(color, theme.FIELD)
        assert ratio >= theme.TEXT_CONTRAST_MIN, (color, ratio)


def test_graphics_on_field_meet_3():
    # 値の弧・帯の中・過負荷・琥珀は地の上で「図形」の基準を満たす。
    for color in (theme.VALUE, theme.BAND_ON, theme.OVER, theme.AMBER):
        ratio = theme.contrast_ratio(color, theme.FIELD)
        assert ratio >= theme.GRAPHIC_CONTRAST_MIN, (color, ratio)


def test_header_title_meets_4_5():
    ratio = theme.contrast_ratio(theme.TEXT, theme.HEADER)
    assert ratio >= theme.TEXT_CONTRAST_MIN


def test_no_green_in_palette():
    # 緑は使わない（constraints.md）。HSL の色相 90〜170° で彩度が 0.25 を超える
    # 色が、役割の色にも図柄の色にも 1 つも無いことを確かめる。
    # 0.25 は「灰色に近い色は色相が意味を持たず、緑には見えない」ための足切り。
    # 彩度 0.20 ほどの灰み（補足 #94a3b8、仮に同じ彩度で色相 145° の #94b8a3 でも）は
    # 灰色に見えるので拾わず、それより彩度の高い「緑と読める色」だけを拾う。
    for name, color in theme.PALETTE.items():
        hue, saturation = theme.hue_saturation(color)
        is_green = 90.0 <= hue <= 170.0 and saturation > 0.25
        assert not is_green, (name, color, hue, saturation)


def test_known_low_contrast_is_listed():
    # 「文字」の基準（4.5）を当てる組。見出しの補足以外は test_text_on_field_meets_4_5
    # で個別に確かめ済みで、ここでは「既知の低コントラストを除けば全部満たす」ことを見る。
    text_pairs = {
        (theme.TEXT, theme.FIELD),
        (theme.SUBTEXT, theme.FIELD),
        (theme.OVER, theme.FIELD),
        (theme.TEXT, theme.HEADER),
        (theme.HEADER_SUB, theme.HEADER),
    }
    # 「図形」の基準（3.0）を当てる組。
    graphic_pairs = {
        (theme.VALUE, theme.FIELD),
        (theme.BAND_ON, theme.FIELD),
        (theme.OVER, theme.FIELD),
        (theme.AMBER, theme.FIELD),
        (theme.BAND, theme.FIELD),
    }

    known = theme.KNOWN_LOW_CONTRAST
    assert set(known.keys()) == {
        (theme.BAND, theme.FIELD),
        (theme.HEADER_SUB, theme.HEADER),
    }
    for pair, info in known.items():
        actual = theme.contrast_ratio(*pair)
        # 記録した値（計算してそのまま書いたもの）が実際の計算と一致すること。
        assert actual == pytest.approx(info["ratio"], abs=0.01)
        assert info["reason"]

    for pair in text_pairs - known.keys():
        assert theme.contrast_ratio(*pair) >= theme.TEXT_CONTRAST_MIN, pair
    for pair in graphic_pairs - known.keys():
        assert theme.contrast_ratio(*pair) >= theme.GRAPHIC_CONTRAST_MIN, pair


@pytest.mark.parametrize("color", ["#fff", "#12345", "#1234567", "", "red"])
def test_non_six_digit_color_is_rejected(color):
    with pytest.raises(ValueError):
        theme.contrast_ratio(color, "#000000")


def test_non_hex_digits_are_rejected():
    with pytest.raises(ValueError):
        theme.relative_luminance("#gggggg")
