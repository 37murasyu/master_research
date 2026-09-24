"""被験者ゲージの書体。Mac にインストール（アクティベート）済みのフォントワークスの書体を使う。

有料の書体は ``.app`` に同梱しない（再配布はライセンスで許されていない。同梱の IPAex ゴシックと
同じ理由で Meiryo も同梱していない。``app/core/resources.py``）。起動した Mac に入っているものを
名前で探し、無ければヒラギノ → 同梱の IPAex ゴシックへ落とす。

書体の組（プリセット）は設定 ``GAUGE_FONT_PRESET`` で選ぶ。文字は 3 つの役に分ける。

- ``heading``: 見出し帯の題（「上肢の仕事量」）
- ``numeral``: 値・回数・帯の両端の数字（等幅数字 ``tnum`` を指定し、30 Hz で値が変わっても揺れないように）
- ``text``: 部位名・状態・凡例などの残り

フォントの名前は、LETS と Adobe Fonts で、また日本語名と英語名で揺れる（例: 「FOT-ロダン Pro EB」
「FOT-Rodin Pro」＋スタイル「EB」）。そこで、空白・記号を除いて小文字にした名前に鍵の文字列が
含まれるかで探し、太さはスタイル名や名前の末尾の記号（EB・B など）で決める。手元でどう見えているかは
``python -m app.gauge.fonts`` で確かめられる。

名前の照合と太さの選び方は Qt に依存しない（試験のため）。Qt の ``QFontDatabase`` を引くのは
``_installed_faces`` と ``make_font`` だけ。
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "Face",
    "FamilySpec",
    "Preset",
    "current_preset",
    "font_generation",
    "make_font",
    "pick_face",
    "role_for_label",
    "set_preset",
]


# ---------------------------------------------------------------------------
# プリセット
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    """探す書体。``keys`` のどれかを名前に含み、``excludes`` のどれも含まないもの。

    鍵は ``_normalize`` と同じ形（空白・記号なし、小文字、NFKC）で書く。
    """

    label: str
    keys: tuple[str, ...]
    excludes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Preset:
    label: str
    heading: tuple[FamilySpec, ...]
    numeral: tuple[FamilySpec, ...]
    text: tuple[FamilySpec, ...]
    # 見つからないときの代わり（見出しが明朝のプリセットは、まずヒラギノ明朝へ落とす）
    heading_fallback: tuple[str, ...] = ()


UD_KAKUGO_LARGE = FamilySpec("UD角ゴ_ラージ", ("ud角ゴラージ", "udkakugolarge"))
RODIN = FamilySpec(
    "ロダン",
    ("ロダンpro", "rodinpro", "ロダン", "rodin"),
    # 別の書体（ニューロダン・ロダンわんぱく など）を拾わない
    excludes=("ニュー", "new", "ntlg", "わんぱく", "wanpaku", "マリア", "maria", "カトレア", "cattleya",
              "ハミング", "humming", "ポップ", "pop", "エクストラ", "rounded"),
)
TSUKUSHI_MIDASHI = FamilySpec("筑紫A見出ミン", ("筑紫a見出ミン", "tsukushiamidashimin", "tsukuamidashimin"))
KAIMIN_SORA = FamilySpec("解ミン 宙", ("解ミン宙", "kaiminsora"))

_MINCHO_FALLBACK = ("Hiragino Mincho ProN", "ヒラギノ明朝 ProN")

PRESETS: dict[str, Preset] = {
    # すべてゴシック。見出しはロダン、数字と文字は UD角ゴ_ラージ（離れて読む用途）
    "rodin": Preset("見出し ロダン／数字・文字 UD角ゴ_ラージ",
                    (RODIN,), (UD_KAKUGO_LARGE,), (UD_KAKUGO_LARGE,)),
    "tsukushi": Preset("見出し 筑紫A見出ミン／数字・文字 UD角ゴ_ラージ",
                       (TSUKUSHI_MIDASHI,), (UD_KAKUGO_LARGE,), (UD_KAKUGO_LARGE,), _MINCHO_FALLBACK),
    "kaimin": Preset("見出し 解ミン 宙／数字・文字 UD角ゴ_ラージ",
                     (KAIMIN_SORA,), (UD_KAKUGO_LARGE,), (UD_KAKUGO_LARGE,), _MINCHO_FALLBACK),
    # 従来どおり（ヒラギノ角ゴ → IPAex ゴシック）
    "system": Preset("ヒラギノ角ゴ（従来）", (), (), ()),
}
DEFAULT_PRESET = "rodin"

ROLES = ("heading", "numeral", "text")
_HEADING_ROLES = frozenset({"header_title"})
_NUMERAL_ROLES = frozenset({"value_text", "header_rep", "band_label"})


def role_for_label(label_role: str) -> str:
    """``scene.Label.role`` から、書体の役（heading / numeral / text）を決める。"""
    if label_role in _HEADING_ROLES:
        return "heading"
    if label_role in _NUMERAL_ROLES:
        return "numeral"
    return "text"


# ---------------------------------------------------------------------------
# 名前の照合と太さ（Qt に依存しない）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Face:
    """インストール済みの 1 書体（QFontDatabase の family と style の組）。"""

    family: str
    style: str
    weight: int  # CSS の 100〜900 の尺度


# フォントワークスの太さの記号（名前の末尾やスタイル名に付く）と、CSS の太さの対応
_WEIGHT_TOKENS = {
    "ul": 100, "el": 200, "l": 300, "r": 400, "regular": 400, "m": 500, "medium": 500,
    "db": 600, "demibold": 600, "semibold": 600, "b": 700, "bold": 700,
    "e": 800, "eb": 800, "extrabold": 800, "h": 900, "heavy": 900, "ub": 900, "black": 900,
}


def _normalize(name: str) -> str:
    """NFKC にして小文字にし、空白・ハイフン・アンダースコア・中黒を除く。"""
    text = unicodedata.normalize("NFKC", name).lower()
    for ch in " 　-_・":
        text = text.replace(ch, "")
    return text


def _weight_from_name(family: str, style: str) -> int | None:
    """名前の最後の語（「FOT-ロダン Pro EB」の EB）、またはスタイル名から太さを読む。

    太さごとに別の family になっている書体は、スタイル名が「Regular」などの飾りのことがあるので、
    名前の末尾を先に見る。
    """
    words = unicodedata.normalize("NFKC", family).replace("-", " ").split()
    if len(words) >= 2 and words[-1].lower() in _WEIGHT_TOKENS:
        return _WEIGHT_TOKENS[words[-1].lower()]
    return _WEIGHT_TOKENS.get(_normalize(style))


def matches(spec: FamilySpec, family: str) -> bool:
    name = _normalize(family)
    return any(key in name for key in spec.keys) and not any(ex in name for ex in spec.excludes)


def pick_face(specs: tuple[FamilySpec, ...], faces: list[Face], weight: int) -> Face | None:
    """``specs`` の順に探し、見つかった書体のうち ``weight`` に一番近い太さのものを返す。

    同じ近さなら太い方（見出しや数字が細る方向に外れないように）。
    """
    for spec in specs:
        hits = [face for face in faces if matches(spec, face.family)]
        if hits:
            return min(hits, key=lambda f: (abs(f.weight - weight), -f.weight))
    return None


# ---------------------------------------------------------------------------
# 今のプリセット（プロセスに 1 つ）
# ---------------------------------------------------------------------------

_preset_name = DEFAULT_PRESET
# プリセットを変えるたびに増える。ゲージの動かない層のキャッシュの鍵に入れる
_generation = 0


def set_preset(name: str | None) -> str:
    """プリセットを切り替え、実際に使う名前を返す。知らない名前・空なら既定に戻す。"""
    global _preset_name, _generation  # pylint: disable=global-statement
    name = (name or "").strip().lower() or DEFAULT_PRESET
    if name not in PRESETS:
        print(f"[ゲージ] 知らない書体の組: {name!r}。{DEFAULT_PRESET!r} を使います"
              f"（選べるもの: {', '.join(PRESETS)}）", file=sys.stderr)
        name = DEFAULT_PRESET
    if name != _preset_name:
        _preset_name = name
        _generation += 1
        _FONT_CACHE.clear()
    return name


def current_preset() -> str:
    return _preset_name


def font_generation() -> int:
    return _generation


# ---------------------------------------------------------------------------
# Qt 側
# ---------------------------------------------------------------------------

_FACES: list[Face] | None = None
_GOTHIC_FALLBACK: tuple[str, ...] | None = None
_FONT_CACHE: dict[tuple[str, int, int], object] = {}


def _qt_weight_to_css(qt_weight: int) -> int:
    # Qt 6 の QFont.Weight は CSS と同じ 100〜900 の尺度
    return int(qt_weight)


def _installed_faces() -> list[Face]:
    """この Mac で使える全書体（family × style）。初回だけ引いてキャッシュする。"""
    global _FACES  # pylint: disable=global-statement
    if _FACES is None:
        from app.core.qt import QtGui

        db = QtGui.QFontDatabase
        faces = []
        for family in db.families():
            for style in db.styles(family) or [""]:
                weight = _weight_from_name(family, style)
                if weight is None:
                    weight = _qt_weight_to_css(db.weight(family, style))
                faces.append(Face(family, style, weight))
        _FACES = faces
    return _FACES


def _gothic_fallback() -> tuple[str, ...]:
    """ヒラギノ角ゴ → 同梱の IPAex ゴシック（初回だけ登録する）。"""
    global _GOTHIC_FALLBACK  # pylint: disable=global-statement
    if _GOTHIC_FALLBACK is None:
        from app.core.qt import QtGui
        from app.core.resources import japanese_font_path

        families = ["Hiragino Sans", "Hiragino Kaku Gothic ProN"]
        font_id = QtGui.QFontDatabase.addApplicationFont(str(japanese_font_path()))
        if font_id != -1:
            families.extend(QtGui.QFontDatabase.applicationFontFamilies(font_id))
        _GOTHIC_FALLBACK = tuple(families)
    return _GOTHIC_FALLBACK


def resolve(role: str, weight: int) -> tuple[Face | None, tuple[str, ...]]:
    """役と太さに合う書体（無ければ None）と、見つからないときの代わりの並び。"""
    preset = PRESETS[_preset_name]
    specs: tuple[FamilySpec, ...] = getattr(preset, role)
    face = pick_face(specs, _installed_faces(), weight) if specs else None
    fallback = (preset.heading_fallback if role == "heading" else ()) + _gothic_fallback()
    return face, fallback


def make_font(role: str, pixel_size: int, weight: int):
    """``QFont`` を作る。同じ（役・大きさ・太さ）は使い回す（毎フレーム作り直さない）。"""
    from app.core.qt import QtGui

    key = (role, pixel_size, weight)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return QtGui.QFont(cached)

    face, fallback = resolve(role, weight)
    font = QtGui.QFont()
    if face is None:
        font.setFamilies(list(fallback))
        font.setWeight(QtGui.QFont.Weight(weight))
    else:
        font.setFamilies([face.family, *fallback])
        if face.style:
            font.setStyleName(face.style)
        # 書体そのものの太さを指定する（違う太さを指定すると Qt が太字を合成してしまう）
        font.setWeight(QtGui.QFont.Weight(face.weight))
    font.setPixelSize(max(1, pixel_size))
    if role == "numeral":
        _enable_tabular_figures(font)
    _FONT_CACHE[key] = QtGui.QFont(font)
    return font


def _enable_tabular_figures(font) -> None:
    """等幅数字（OpenType の tnum）。QFont.setFeature は Qt 6.7 から。無ければ何もしない。"""
    from app.core.qt import QtGui

    try:
        font.setFeature(QtGui.QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# 診断: python -m app.gauge.fonts
# ---------------------------------------------------------------------------


def _report() -> str:
    lines = []
    faces = _installed_faces()
    for name, preset in PRESETS.items():
        mark = "（今の設定）" if name == _preset_name else ""
        lines.append(f"■ {name}: {preset.label}{mark}")
        for role, weight in (("heading", 800), ("numeral", 800), ("text", 700)):
            specs = getattr(preset, role)
            face = pick_face(specs, faces, weight) if specs else None
            if face is not None:
                found = f"{face.family} / {face.style or '-'}（太さ {face.weight}）"
            elif specs:
                found = f"見つからない → {', '.join((preset.heading_fallback if role == 'heading' else ()) + _gothic_fallback()[:1])} へ"
            else:
                found = "ヒラギノ角ゴ → IPAex（従来）"
            lines.append(f"    {role:<8} {found}")
    specs = {spec for preset in PRESETS.values() for spec in (*preset.heading, *preset.numeral, *preset.text)}
    fontworks = sorted({
        f.family for f in faces
        if _normalize(f.family).startswith("fot") or any(matches(spec, f.family) for spec in specs)
    })
    lines.append("")
    lines.append(f"この Mac のフォントワークスの書体（FOT- で始まるか、上の組の書体の名前を含むもの）: {len(fontworks)} 件")
    lines.extend(f"    {family}" for family in fontworks)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from app.core.qt import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        set_preset(args[0])
    print(_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
