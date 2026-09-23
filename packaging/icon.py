"""アプリアイコンを生成する（SVG → PNG 各サイズ → .icns）。

    python packaging/icon.py [出力ディレクトリ]      既定: build/icon

**コンセプト: 車椅子を正面から見るとバーベルになり、その上でゲージが今の 1 回を測っている**

- 主題は座位プッシュアップ（``push_up_model.py``）。アームレストを両手で押して体を座面から
  浮かせる、車椅子の上でできる筋トレで、アプリは反復ごとの関節トルクと仕事を測り、
  被験者に扇形のゲージで見せる。
- アイコンはそのゲージ画面の縮図にする。地の濃紺はゲージ画面と同じ。人物の上に扇形のメーターが
  かぶさり、白い弧が目標帯（外側の明るい青の弧）の中まで満ちている。先端の琥珀の印が「今」の値、
  白い弧の途中の切れ目が「前回」の値で、前回から今へ値が伸びてきた＝実時間で測っていることを、
  飾りの動線ではなくゲージそのものの要素で示す（琥珀＝動いているもの、の役割どおり）。
- 正面から見ると、左右の車輪は縦長の板になり、車軸は外へ突き出す。プレートを両端に付けた
  バーベルと同じ形で、座っている人はそのシャフトの位置にいる。負荷を 1RM 法
  （``rm_method.csv``）で評価する＝レジスタンストレーニングだということを、形そのものが言う。
  プレートの赤は IWF 規格の 25 kg。車輪の傾き（キャンバー）は競技用車椅子のもの。
- 造形の文法は 1964 年東京・1972 年ミュンヘンの競技ピクトグラム。均一な太さの線・丸い端・
  単色の平面で、競技を幾何学の身体として描いた系譜に乗せる。
- 国際シンボルマーク（1968）は「座っている人」を描いた。このアイコンは同じ座位の人が
  **座面から浮いている**瞬間を描く。膝と座面の隙間が、この運動そのもの。
- **身体は頭から足先まで一続きに描く。** 胴と脚を切り離すと、身体が分断されて見える。
  脊髄損傷のある人に向けた表現として不適切なので、胴・膝・すねをつなぎ、座面はその奥を通す。
- 椅子は淡い青で人物（白）より一段下げ、身体と道具を見分けられるようにする。

色はゲージ画面と共通: 地の濃紺 #0b1633、溝 #1e293b、目標帯 #60a5fa、値の弧は白、今の印は琥珀 #fbbf24、
プレートの赤 #dc2626。弧の値（今・前回・目標帯の位置）は説明用の固定値で、実測ではない。

背景は単色にする。部品の重なりには背景色の太線で「切り欠き」を入れて前後を分けるが、
QtSvg は水平な線（外接矩形の高さが 0）にグラデーションを塗ると終端の 1 色になり、
切り欠きが縁取りとして浮いてしまう。

小さいサイズ（16・32 px）では前回の切れ目とプレートの段差が潰れるので、それらを省き線を太くした
別図を使う。macOS の iconset はサイズごとに別の画像を持てる。

座標は 1024 px のキャンバス（y は下向き）、左右対称で中心は x=512。macOS のアイコン格子に
合わせ、本体は 100〜924 の角丸四角。描画には PySide6 の QtSvg を使う（SVG Tiny 1.2 相当。
フィルタは使えないので影は重ね描きで作る）。
年代・経緯は一般的な記述に拠っており、対外的な資料に使うなら一次資料で裏を取ること。
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

CANVAS = 1024
BODY_MIN, BODY_MAX = 100, 924
CORNER = 185
CX = CANVAS / 2

# 配色（ゲージ画面と共通）
BG = "#0b1633"
FIGURE = "#ffffff"
CHAIR = "#bfdbfe"
PLATE = "#dc2626"
PLATE_DARK = "#991b1b"
SLEEVE = "#e5e7eb"
TRACK = "#1e293b"          # ゲージの溝
BAND = "#60a5fa"           # 目標帯（値が帯の中にあるときの明るい青）
NOW = "#fbbf24"            # 今の値の印（琥珀＝動いているもの）

# --- 扇形のゲージ（左端が 0、右端が最大。f は 0〜1 の割合） --------------------
GAUGE_CX, GAUGE_CY = 512.0, 600.0
GAUGE_R = 318.0            # 値の弧の中心線の半径
GAUGE_W = 78.0             # 値の弧の太さ（簡略図では太くする）
BAND_OFFSET, BAND_W = 22.0, 22.0   # 目標帯の弧: 値の弧の外側の隙間と太さ
BAND_LO, BAND_HI = 0.24, 0.80      # 目標帯の位置
F_NOW, F_PREV = 0.62, 0.50         # 今と前回（どちらも帯の中。前回→今で伸びている）

# 人物は縮小して扇の内側に置く（身体の座標は下の定数のまま、全体を拡大縮小する）
FIGURE_SCALE, FIGURE_TOP = 0.54, 318.0

# --- 車椅子（正面）。x は中心からの距離 ---------------------------------------
PLATE_X = 196.0
PLATE_W = 104.0
PLATE_TOP, PLATE_BOTTOM = 452.0, 852.0
CAMBER_DEG = 7.0           # 下端が外へ開く
SLEEVE_LEN = 58.0          # 車軸（＝スリーブ）がプレートの外へ突き出す長さ
SEAT_Y = 626.0
SEAT_HALF = 99.0           # 切り欠きがプレートに食い込まない長さ（プレート内縁 − 切り欠き − 部材の半幅）
ARMREST_Y = 452.0
ARMREST_HALF = 46.0
FOOTREST_Y, FOOTREST_HALF = 846.0, 120.0

# --- 人物: プッシュアップの最上位（肘を伸ばし切り、腰が座面から浮いている） ----
HEAD_Y, HEAD_R = 196.0, 64.0
SHOULDER_Y, SHOULDER_X = 292.0, 150.0
TORSO_TOP_Y, TORSO_TOP_HALF = 302.0, 100.0
TORSO_BOTTOM_Y, TORSO_BOTTOM_HALF = 520.0, 46.0   # 逆三角形の胴
HAND_Y = 420.0             # アームレストのパッドの上
# 正面から見た座位では太ももが手前を向くので、膝を横長の形（LAP）で描き、胴とつなぐ。
# すねは膝から足先まで。頭から足先までが一続きになる。膝と座面の間の隙間が「浮き」
LAP_Y, LAP_HALF = 548.0, 60.0
FOOT_X, FOOT_Y = 62.0, 812.0

LIMB = 64.0                # 手足の太さ
CHAIR_W = 34.0             # 椅子の部材の太さ
GAP = 22.0                 # 重なりに入れる切り欠きの幅


def _pts(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _stroke(points, color: str, width: float) -> str:
    return (
        f'<polyline points="{_pts(points)}" fill="none" stroke="{color}" stroke-width="{width:.0f}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _knocked(strokes, color: str, width: float, gap: float) -> list[str]:
    """切り欠き付きで線を引く。背景色で太く引いてから本体を重ねる。"""
    return [_stroke(p, BG, width + 2 * gap) for p in strokes] + [_stroke(p, color, width) for p in strokes]


def _background() -> str:
    size = BODY_MAX - BODY_MIN
    return (
        f'<rect x="{BODY_MIN}" y="{BODY_MIN + 10}" width="{size}" height="{size}" rx="{CORNER}" '
        f'fill="#000000" fill-opacity="0.22"/>\n  '
        f'<rect x="{BODY_MIN}" y="{BODY_MIN}" width="{size}" height="{size}" rx="{CORNER}" fill="{BG}"/>'
    )


def _plate(side: int, detailed: bool) -> str:
    """片側の車輪＝プレート（縁から見た姿）と、外へ突き出す車軸＝スリーブ。side は -1 左 / +1 右。"""
    x = CX + side * PLATE_X
    cy = (PLATE_TOP + PLATE_BOTTOM) / 2
    h = PLATE_BOTTOM - PLATE_TOP
    # SVG の rotate は正が時計回り。左の車輪は上端を内へ（時計回り）倒す
    parts = [
        f'<g transform="rotate({-side * CAMBER_DEG} {x} {cy})">',
        f'<rect x="{x - PLATE_W / 2}" y="{PLATE_TOP}" width="{PLATE_W}" height="{h}" '
        f'rx="{PLATE_W / 2 - 6}" fill="{PLATE}"/>',
    ]
    if detailed:
        # バンパープレートの段差（縁より一段低い面）
        parts.append(
            f'<rect x="{x - PLATE_W / 2 + 22}" y="{PLATE_TOP + 40}" width="{PLATE_W - 44}" height="{h - 80}" '
            f'rx="{PLATE_W / 2 - 28}" fill="{PLATE_DARK}"/>'
        )
    parts.append("</g>")
    inner = x + side * (PLATE_W / 2 - 10)
    outer = x + side * (PLATE_W / 2 + SLEEVE_LEN)
    parts.append(_stroke([(inner, cy), (outer, cy)], SLEEVE, 40))
    return "\n  ".join(parts)


def _seat(width: float) -> str:
    """座面。人物の切り欠きの後・本体の前に描き、脚の奥を通って見えるようにする（脚を切らない）。"""
    return _stroke([(CX - SEAT_HALF, SEAT_Y), (CX + SEAT_HALF, SEAT_Y)], CHAIR, width)


def _chair(width: float, gap: float) -> list[str]:
    armrests = [
        [(CX + side * PLATE_X - ARMREST_HALF, ARMREST_Y), (CX + side * PLATE_X + ARMREST_HALF, ARMREST_Y)]
        for side in (-1, 1)
    ]
    footrest = [(CX - FOOTREST_HALF, FOOTREST_Y), (CX + FOOTREST_HALF, FOOTREST_Y)]
    return _knocked([*armrests, footrest], CHAIR, width, gap)


def _figure(limb: float, gap: float, behind: str = "") -> list[str]:
    """人物。``behind`` は切り欠きと本体の間に描くもの（人物の奥に見える座面）。"""
    torso = _pts([
        (CX - TORSO_TOP_HALF, TORSO_TOP_Y), (CX + TORSO_TOP_HALF, TORSO_TOP_Y),
        (CX + TORSO_BOTTOM_HALF, TORSO_BOTTOM_Y), (CX - TORSO_BOTTOM_HALF, TORSO_BOTTOM_Y),
    ])
    strokes = [
        [(CX - SHOULDER_X, SHOULDER_Y), (CX + SHOULDER_X, SHOULDER_Y)],
        *[[(CX + s * SHOULDER_X, SHOULDER_Y), (CX + s * PLATE_X, HAND_Y)] for s in (-1, 1)],
        [(CX - LAP_HALF, LAP_Y), (CX + LAP_HALF, LAP_Y)],
        *[[(CX + s * LAP_HALF, LAP_Y), (CX + s * FOOT_X, FOOT_Y)] for s in (-1, 1)],
    ]
    out = []
    # 1 周目で切り欠き（背景色・太め）、2 周目で本体
    for color, width, head_r in ((BG, limb + 2 * gap, HEAD_R + gap), (FIGURE, limb, HEAD_R)):
        out.append(
            f'<polygon points="{torso}" fill="{color}" stroke="{color}" stroke-width="{width:.0f}" '
            f'stroke-linejoin="round"/>'
        )
        out += [_stroke(p, color, width) for p in strokes]
        out.append(f'<circle cx="{CX}" cy="{HEAD_Y}" r="{head_r}" fill="{color}"/>')
        if color == BG and behind:
            out.append(behind)
    return out


def _gauge_point(radius: float, f: float) -> tuple[float, float]:
    """扇の上の点。f=0 が左端、f=1 が右端（上を通る半円）。"""
    angle = math.radians(180 + 180 * f)
    return GAUGE_CX + radius * math.cos(angle), GAUGE_CY + radius * math.sin(angle)


def _gauge_arc(radius: float, f0: float, f1: float, color: str, width: float) -> str:
    (x0, y0), (x1, y1) = _gauge_point(radius, f0), _gauge_point(radius, f1)
    # 端は平ら（丸めると帯の境界と値の位置がずれて見える。ゲージ画面と同じ）
    return (
        f'<path d="M{x0:.1f} {y0:.1f}A{radius} {radius} 0 0 1 {x1:.1f} {y1:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="{width:.0f}"/>'
    )


def _gauge_tick(f: float, overhang: float, color: str, width: float) -> str:
    """値の弧を横切る短い線（前回の切れ目・今の印）。"""
    half = GAUGE_W / 2 + overhang
    (x0, y0), (x1, y1) = _gauge_point(GAUGE_R - half, f), _gauge_point(GAUGE_R + half, f)
    return (
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" '
        f'stroke-width="{width:.0f}" stroke-linecap="round"/>'
    )


def _gauge(detailed: bool) -> list[str]:
    width = GAUGE_W if detailed else GAUGE_W * 1.25
    band_w = BAND_W if detailed else BAND_W * 1.4
    band_r = GAUGE_R + width / 2 + BAND_OFFSET
    parts = [
        _gauge_arc(GAUGE_R, 0.0, 1.0, TRACK, width),
        _gauge_arc(band_r, BAND_LO, BAND_HI, BAND, band_w),
        _gauge_arc(GAUGE_R, 0.0, F_NOW, FIGURE, width),
    ]
    if detailed:
        parts.append(_gauge_tick(F_PREV, 6, BG, 12))  # 前回: 白い弧の切れ目
    parts.append(_gauge_tick(F_NOW, 14, NOW, 26 if detailed else 34))
    return parts


def build_svg(detailed: bool = True) -> str:
    """アイコンの SVG。``detailed=False`` は 16/32 px 用の簡略図（線を太く、細部を省く）。"""
    k = 1.0 if detailed else 1.25
    body = [_background(), *_gauge(detailed)]
    person = [_plate(-1, detailed), _plate(+1, detailed)]
    person += _chair(CHAIR_W * k, GAP * k)
    person += _figure(LIMB * k, GAP * k, behind=_seat(CHAIR_W * k))
    shift = CX * (1 - FIGURE_SCALE)
    body.append(
        f'<g transform="translate({shift:.1f},{FIGURE_TOP:.1f}) scale({FIGURE_SCALE})">\n  '
        + "\n  ".join(person)
        + "\n  </g>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" '
        f'width="{CANVAS}" height="{CANVAS}" viewBox="0 0 {CANVAS} {CANVAS}">\n  '
        + "\n  ".join(body)
        + "\n</svg>\n"
    )


# --- 書き出し ---------------------------------------------------------------
# iconset の命名規則: icon_<pt>x<pt>[@2x].png。(ファイル名, 画素数, 簡略図か)
ICONSET = [
    ("icon_16x16.png", 16, True),
    ("icon_16x16@2x.png", 32, True),
    ("icon_32x32.png", 32, True),
    ("icon_32x32@2x.png", 64, False),
    ("icon_128x128.png", 128, False),
    ("icon_128x128@2x.png", 256, False),
    ("icon_256x256.png", 256, False),
    ("icon_256x256@2x.png", 512, False),
    ("icon_512x512.png", 512, False),
    ("icon_512x512@2x.png", 1024, False),
]


def render_png(svg: str, size: int, out: Path) -> None:
    from PySide6 import QtCore, QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    if not renderer.isValid():
        raise ValueError("SVG を解釈できません")
    image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    if not image.save(str(out)):
        raise OSError(f"PNG を保存できません: {out}")


def build_icns(out_dir: Path, name: str = "AppIcon") -> Path:
    """SVG 2 種と iconset を書き出し、iconutil で .icns にまとめる。"""
    from PySide6 import QtGui

    # QImage の描画には QGuiApplication が要る。画面は使わないので offscreen で足りる
    _app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(["icon", "-platform", "offscreen"])

    out_dir.mkdir(parents=True, exist_ok=True)
    svgs = {False: build_svg(detailed=True), True: build_svg(detailed=False)}
    (out_dir / f"{name}.svg").write_text(svgs[False], encoding="utf-8")
    (out_dir / f"{name}_small.svg").write_text(svgs[True], encoding="utf-8")

    iconset = out_dir / f"{name}.iconset"
    iconset.mkdir(exist_ok=True)
    for filename, size, small in ICONSET:
        render_png(svgs[small], size, iconset / filename)

    icns = out_dir / f"{name}.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    return icns


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/icon")
    print(build_icns(target))
