"""被験者ゲージの人物ピクトグラム（SVG）。

正本は画面案の生成スクリプト ``mk_subject.py`` の ``FIG``・``ICON``（最新版）。
そちらは CSS の class と ``var(--x)`` で色を持つが、QtSvg（SVG Tiny 1.2 相当）は
CSS の class セレクタも ``var()`` も解さないので、ここでは形（座標・太さ・
描く順）だけを ``FIG``・``ICON`` からそのまま移し、色は関数の引数を各要素の
属性に直書きする。

``figure_svg`` が描く人物は、頭から足先まで一続きに描く（胴と膝とすねをつなぐ）。
これは constraints.md の「人物」の節が絶対に守ると定めているところで、
脊髄損傷のある人に向けた配慮として身体を分断して見せないため。描く順は
「椅子のパッドとプレートとスリーブ → 脚の切り欠き（地の色）→ 座面 →
頭・肩・胴・腕 → 膝・すね」（同じ節のとおり）で、座面は人物の奥を通るように
人物より先に描く。
"""

from __future__ import annotations

__all__ = ["figure_svg", "header_icon_svg"]


def figure_svg(
    *,
    figure: str,
    chair: str,
    plate: str,
    plate_dark: str,
    sleeve: str,
    background: str,
) -> str:
    """車椅子に座る人物の絵。viewBox は 800×450（ゲージ画面と同じ設計座標）。

    引数はすべて役割の色（``app/shell/theme.py``）をそのまま渡す想定:
    ``figure`` は人物そのもの、``chair`` は椅子の部材、``plate``・``plate_dark``
    はプレートの明暗、``sleeve`` は車軸のスリーブ、``background`` は地の色
    （脚の切り欠きに使う。切り欠きは「消しゴムで消す」のではなく、地の色で
    上から塗って前後を分ける、という mk_subject.py の描き方をそのまま踏襲する）。
    """
    elements = [
        f'<path id="chair-pads" d="M322 284H350M450 284H478" '
        f'fill="none" stroke="{chair}" stroke-width="8" stroke-linecap="round"/>',
        f'<g id="plate-l" transform="rotate(7 332 352)">'
        f'<rect x="318" y="292" width="28" height="120" rx="12" fill="{plate}"/>'
        f'<rect x="324" y="304" width="16" height="96" rx="7" fill="{plate_dark}"/>'
        f'</g>',
        f'<g id="plate-r" transform="rotate(-7 468 352)">'
        f'<rect x="454" y="292" width="28" height="120" rx="12" fill="{plate}"/>'
        f'<rect x="460" y="304" width="16" height="96" rx="7" fill="{plate_dark}"/>'
        f'</g>',
        f'<path id="sleeves" d="M300 352H318M482 352H500" '
        f'fill="none" stroke="{sleeve}" stroke-width="10" stroke-linecap="round"/>',
        f'<path id="leg-knockout" d="M386 268L387 392M414 268L413 392" '
        f'fill="none" stroke="{background}" stroke-width="22" stroke-linecap="round"/>',
        f'<path id="seat" d="M370 302H430" '
        f'fill="none" stroke="{chair}" stroke-width="8" stroke-linecap="round"/>',
        f'<circle id="head" cx="400" cy="118" r="17" fill="{figure}"/>',
        f'<path id="shoulders" d="M362 150H438" '
        f'fill="none" stroke="{figure}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<path id="torso" d="M372 150H428L416 256H384Z" '
        f'fill="{figure}" stroke="{figure}" stroke-width="10" stroke-linejoin="round"/>',
        f'<path id="arms" d="M362 150L336 272M438 150L464 272" '
        f'fill="none" stroke="{figure}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<path id="knees" d="M386 266H414" '
        f'fill="none" stroke="{figure}" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<path id="shins" d="M386 266L387 392M414 266L413 392" '
        f'fill="none" stroke="{figure}" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>',
    ]
    body = "\n".join(elements)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" '
        'viewBox="0 0 800 450">\n'
        f"{body}\n"
        "</svg>\n"
    )


def header_icon_svg(*, figure: str, plate: str) -> str:
    """見出し帯の小さいアイコン（車椅子を正面から見た形）。viewBox は 48×48。

    mk_subject.py の ``ICON`` をそのまま移す。元は色が ``"#fff"``（人物・線）と
    ``"#dc2626"``（プレート）の固定値だったところを、引数の属性に置き換える。
    """
    body = (
        '<g transform="translate(18,10) scale(0.92)" fill="none" '
        f'stroke="{figure}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">'
        f'<rect x="5" y="23" width="6" height="21" rx="3" fill="{plate}" stroke="none" '
        f'transform="rotate(7 8 33)"/>'
        f'<rect x="37" y="23" width="6" height="21" rx="3" fill="{plate}" stroke="none" '
        f'transform="rotate(-7 40 33)"/>'
        f'<circle cx="24" cy="8" r="4.5" fill="{figure}" stroke="none"/>'
        '<path d="M15 17h18M15 17l-6 7M33 17l6 7"/>'
        f'<path d="M18 17h12l-2.5 12h-7z" fill="{figure}"/>'
        '<path d="M21 31h6M21 31v12M27 31v12"/>'
        "</g>"
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" '
        'viewBox="0 0 48 48">\n'
        f"{body}\n"
        "</svg>\n"
    )
