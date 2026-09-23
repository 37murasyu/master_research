"""日本語テキストの描画（``utils.put_text_jp`` / ``utils.draw_text_jp``）を固定する。

**なぜこのテストがあるか。**

文字の描画が 2 か所にあった（KNOWN_ISSUES §4-2）。``utils.put_text_jp`` は毎回フレーム全体を
PIL へ往復変換し、``master_research_code.py`` はそれを避けるため文字ごとのスプライトを
キャッシュして合成する ``_blit_label`` を別に持っていた。折り返し（常に折り返すか、長いときだけか）と
行送り（PIL の「A の高さ + 4」か、文字サイズ × 1.25 か）が食い違い、キャリブレーション画面の
2 行目がずれていた。

いまはスプライト合成を ``utils`` に移して一本化した。仕様は従来の ``put_text_jp``（PIL で描く）で、
期待値もその手順で直接描いて作る。
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.core.resources import japanese_font
from utils import draw_text_jp, put_text_jp

WHITE = (255, 255, 255)


def _pil_reference(img, text, position, font_size, color, line_width):
    """従来の put_text_jp と同じ手順（PIL で直接描く）。"""
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(position, textwrap.fill(text, width=line_width),
                             font=japanese_font(int(font_size)), fill=color)
    return np.array(pil)


def _canvas(value=0, shape=(120, 420, 3)):
    return np.full(shape, value, dtype=np.uint8)


class TestPutTextJp:
    def test_returns_a_new_array_and_leaves_the_input_untouched(self):
        img = _canvas()
        out = put_text_jp(img, "右手首 E:12.3", (10, 10), 24, WHITE, 20)
        assert out is not img
        assert not img.any(), "入力の画像が書き換えられた（新しい配列を返す約束）"
        assert out.any()

    def test_a_label_matches_pil(self):
        img = _canvas()
        out = put_text_jp(img, "右手首 E:12.3", (10, 10), 24, WHITE, 20)
        np.testing.assert_array_equal(out, _pil_reference(img, "右手首 E:12.3", (10, 10), 24, WHITE, 20))

    def test_on_grey_the_difference_is_rounding_only(self):
        img = _canvas(128)
        out = put_text_jp(img, "左肘 E:0.5", (10, 10), 24, WHITE, 20)
        ref = _pil_reference(img, "左肘 E:0.5", (10, 10), 24, WHITE, 20)
        assert np.max(np.abs(out.astype(int) - ref.astype(int))) <= 1

    def test_long_text_wraps_like_pil(self):
        """20 字を超える文は折り返し、2 行目の位置も PIL と同じ（キャリブレーション画面の説明文）。"""
        text = "チェッカーボードを両方のカメラに写してからスペースキーを押してください"
        img = _canvas(shape=(160, 560, 3))
        out = put_text_jp(img, text, (10, 10), 24, WHITE, 20)
        ref = _pil_reference(img, text, (10, 10), 24, WHITE, 20)
        rows_out = np.nonzero(out.any(axis=(1, 2)))[0]
        rows_ref = np.nonzero(ref.any(axis=(1, 2)))[0]
        np.testing.assert_array_equal(rows_out, rows_ref, err_msg="行送りが PIL と違う")
        assert np.max(np.abs(out.astype(int) - ref.astype(int))) <= 1

    def test_a_short_newline_is_folded_like_pil(self):
        """textwrap.fill は改行を空白に置き換えるので、短い文の改行も 1 行になる（従来どおり）。"""
        img = _canvas()
        out = put_text_jp(img, "A\nB", (10, 10), 24, WHITE, 20)
        np.testing.assert_array_equal(out, _pil_reference(img, "A\nB", (10, 10), 24, WHITE, 20))

    def test_grayscale_images_are_still_drawn(self):
        img = np.zeros((60, 200), dtype=np.uint8)
        out = put_text_jp(img, "肩", (10, 10), 24, 255, 20)
        assert out.shape == img.shape and out.any()


class TestDrawTextJp:
    def test_draws_in_place(self):
        img = _canvas()
        result = draw_text_jp(img, "右肩 E:1.0", (10, 10), 24, WHITE)
        assert result is img
        assert img.any()

    def test_text_running_off_the_edge_is_clipped(self):
        img = _canvas(shape=(40, 60, 3))
        draw_text_jp(img, "右手首 E:12.3", (30, 20), 24, WHITE)
        assert img.any()

    @pytest.mark.parametrize("colour", [(255, 0, 0), (0, 128, 255)])
    def test_colour_channels_follow_the_array_order(self, colour):
        """色は配列のチャネル順（OpenCV の画像なら BGR）でそのまま書く。PIL と同じ。"""
        img = _canvas()
        out = put_text_jp(img, "肘", (10, 10), 24, colour, 20)
        np.testing.assert_array_equal(out, _pil_reference(img, "肘", (10, 10), 24, colour, 20))
