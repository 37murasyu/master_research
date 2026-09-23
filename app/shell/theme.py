"""被験者ゲージの色の役割とコントラストの検査。Qt には依存しない。

このモジュールは、この後の scene（被験者ゲージの描く場面）・widget・計測画面の
部品が色を取り出す唯一の場所になる。Qt に依存しないのは、色の役割そのものは
描画の道具（QPainter 等）と無関係な決めごとであり、この試験もそうであるように
Qt を起動せずに検査できるようにするため。

色は「1 色に 1 つの役割」だけを持つ（constraints.md の「色」の節が正本）：
青 = 操作・現在地／目標帯、琥珀 = 動いているもの、赤 = 異常・過負荷、無彩色 = その他。
プレート（人物の絵の座面プレート）の赤は、この役割の赤とは別物（「異常」を示す
色ではなく、ただの図柄の色）なので、役割の名前（``OVER``）とは別に
``PLATE``・``PLATE_DARK``・``SLEEVE`` という名前で持ち、混同を避ける。

コントラストは WCAG 2.x の相対輝度（sRGB の各チャンネルを線形化してから
加重和を取る）と、コントラスト比 ``(L1 + 0.05) / (L2 + 0.05)``（L1 は明るい方の
相対輝度）で計算する。基準は、細い文字は見えにくいので「文字」が 4.5、
太く大きく描く図形は「図形」が 3.0（WCAG の Non-text Contrast に合わせた値）。

この基準に満たない組が 2 つあるが、色は constraints.md の正本のまま変えない。
理由は ``KNOWN_LOW_CONTRAST`` に書く。
"""

from __future__ import annotations

import colorsys

__all__ = [
    "FIELD",
    "TRACK",
    "BAND",
    "BAND_ON",
    "VALUE",
    "OVER",
    "TEXT",
    "SUBTEXT",
    "HEADER",
    "HEADER_SUB",
    "AMBER",
    "CHAIR",
    "PLATE",
    "PLATE_DARK",
    "SLEEVE",
    "PALETTE",
    "TEXT_CONTRAST_MIN",
    "GRAPHIC_CONTRAST_MIN",
    "relative_luminance",
    "contrast_ratio",
    "hue_saturation",
    "KNOWN_LOW_CONTRAST",
]

# --- 役割の色（1 色に 1 つの役割。constraints.md の「色」の節が正本）--------

FIELD = "#0b1633"  # 地
TRACK = "#1e293b"  # 溝
BAND = "#1d4ed8"  # 帯（目標帯の外枠。青＝操作・現在地／目標帯）
BAND_ON = "#60a5fa"  # 帯の中（今、目標帯に入っている）
VALUE = "#ffffff"  # 値の弧（無彩色＝その他）
OVER = "#ef4444"  # 過負荷（赤＝異常・過負荷）
TEXT = "#ffffff"  # 文字
SUBTEXT = "#94a3b8"  # 補足
HEADER = "#2563eb"  # 見出し帯
HEADER_SUB = "#dbeafe"  # 見出しの補足
AMBER = "#fbbf24"  # 琥珀（動いているもの）
CHAIR = "#93c5fd"  # 椅子

# --- 図柄の色（役割の色ではない。人物の絵にだけ使う。名前を分けて混同を避ける）--

PLATE = "#dc2626"  # プレート（明）
PLATE_DARK = "#991b1b"  # プレート（暗）
SLEEVE = "#e5e7eb"  # スリーブ

# 「1 色に 1 つの役割」「緑を使わない」を機械的に確かめられるように、
# 名前と 16 進値をまとめておく（役割の色・図柄の色の両方を含む）。
PALETTE: dict[str, str] = {
    "FIELD": FIELD,
    "TRACK": TRACK,
    "BAND": BAND,
    "BAND_ON": BAND_ON,
    "VALUE": VALUE,
    "OVER": OVER,
    "TEXT": TEXT,
    "SUBTEXT": SUBTEXT,
    "HEADER": HEADER,
    "HEADER_SUB": HEADER_SUB,
    "AMBER": AMBER,
    "CHAIR": CHAIR,
    "PLATE": PLATE,
    "PLATE_DARK": PLATE_DARK,
    "SLEEVE": SLEEVE,
}


# --- コントラスト（WCAG 2.x）------------------------------------------------

TEXT_CONTRAST_MIN = 4.5  # 「文字」の基準
GRAPHIC_CONTRAST_MIN = 3.0  # 「図形」の基準（Non-text Contrast）


def _linearize(channel: float) -> float:
    """sRGB の 1 チャンネル（0〜1）を、相対輝度の計算用に線形化する。"""
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _to_rgb01(color: str) -> tuple[float, float, float]:
    """``"#rrggbb"`` を 0〜1 の (r, g, b) に変える。"""
    hex_part = color.lstrip("#")
    if len(hex_part) != 6:
        raise ValueError(f"6桁の16進カラーコードではない: {color!r}")
    r = int(hex_part[0:2], 16) / 255
    g = int(hex_part[2:4], 16) / 255
    b = int(hex_part[4:6], 16) / 255
    return r, g, b


def relative_luminance(color: str) -> float:
    """WCAG の相対輝度（0〜1）。sRGB を線形化してから加重和を取る。"""
    r, g, b = _to_rgb01(color)
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG のコントラスト比 ``(L1 + 0.05) / (L2 + 0.05)``（L1 が明るい方の相対輝度）。

    引数の順序によらず同じ値になる（明暗はここで決める）。白と黒で 21、
    同じ色どうしで 1 になる。
    """
    luminance_a, luminance_b = relative_luminance(a), relative_luminance(b)
    lighter, darker = max(luminance_a, luminance_b), min(luminance_a, luminance_b)
    return (lighter + 0.05) / (darker + 0.05)


def hue_saturation(color: str) -> tuple[float, float]:
    """HSL の色相（度、0〜360）と彩度（0〜1）。緑が紛れていないかの検査に使う。"""
    r, g, b = _to_rgb01(color)
    hue, _lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return hue * 360.0, saturation


# --- 基準に満たない既知の組（色は変えない。理由をここに書く）----------------
#
# 値は実際に contrast_ratio で計算した結果をそのまま書く（動的に計算し直さない）。
# 計算のしかたを後で変えても、ここに記録した「今の色で何倍だったか」は
# 変わらないので、監査の記録として独立させておく（test_known_low_contrast_is_listed
# が、この値と実際の計算が一致することを確かめる）。

KNOWN_LOW_CONTRAST: dict[tuple[str, str], dict[str, object]] = {
    (BAND, FIELD): {
        "ratio": 2.66,  # 実測 約2.6628（図形の基準 3.0 に届かない）
        "reason": (
            "帯（目標帯の外枠）は地の上に太く大きく描く輪であり、細い文字や"
            "小さな記号ではない。基準（3.0）にわずかに届かないが、実物では"
            "内側の帯の中（BAND_ON）や溝（TRACK）と並ぶことで境界がわかる。"
            "色は constraints.md の正本のまま変えない。"
        ),
    },
    (HEADER_SUB, HEADER): {
        "ratio": 4.24,  # 実測 約4.2363（文字の基準 4.5 に届かない）
        "reason": (
            "見出しの補足文字は見出し帯の上に添える小さな一行で、見出し本体の"
            "文字（TEXT。HEADER の上で 4.5 を満たす）ではない。基準にわずかに"
            "届かないが、色は constraints.md の正本のまま変えない。"
        ),
    },
}
