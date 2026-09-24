"""被験者ゲージの書体。Mac にインストール（アクティベート）済みのフォントワークスの書体を使う。

有料の書体は ``.app`` に同梱しない（再配布はライセンスで許されていない。同梱の IPAex ゴシックと
同じ理由で Meiryo も同梱していない。``app/core/resources.py``）。起動した Mac に入っているものを
名前で探し、無ければヒラギノ → 同梱の IPAex ゴシックへ落とす。

書体の組（プリセット）は設定 ``GAUGE_FONT_PRESET`` で選ぶ。組は ``FontSet`` にして窓（``GaugeWidget``）が
持つ（``GaugeWindow(font_preset=...)``・``begin(font_preset=...)``。同じ組は ``font_set`` で使い回す）。
文字は 3 つの役に分ける。

- ``heading``: 見出し帯の題（「上肢の仕事量」）
- ``numeral``: 値・回数・帯の両端の数字（等幅数字 ``tnum`` を指定し、30 Hz で値が変わっても揺れないように）
- ``text``: 部位名・状態・凡例などの残り

フォントの名前は、LETS と Adobe Fonts で、また日本語名と英語名で揺れる（例: 「FOT-ロダン Pro EB」
「FOT-Rodin Pro」＋スタイル「EB」）。そこで、空白・記号を除いて小文字にした名前に鍵の文字列が
含まれるかで探し、太さはスタイル名や名前の末尾の記号（EB・B など）で決める。手元でどう見えているかは
``python -m app.gauge.fonts [組の名前]`` で確かめられる。

名前の照合と太さの選び方は Qt に依存しない（試験のため）。Qt を使うのは ``_installed_faces``・
``_gothic_fallback``・``FontSet`` の ``QFont`` を作るところだけ（どれも呼ばれたときに import する）。
"""

from __future__ import annotations

import functools
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "Face",
    "FamilySpec",
    "FontSet",
    "Preset",
    "font_set",
    "pick_face",
    "role_for_label",
]


# ---------------------------------------------------------------------------
# プリセット
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    """探す書体。``keys`` のどれかを名前に含み、``excludes`` のどれも含まないもの。

    鍵は ``_normalize`` と同じ形（空白・記号なし、小文字、NFKC）で書く。
    """

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


UD_KAKUGO_LARGE = FamilySpec(("ud角ゴラージ", "udkakugolarge"))
RODIN = FamilySpec(
    ("ロダンpro", "rodinpro", "ロダン", "rodin"),
    # 別の書体（ニューロダン・ロダンわんぱく など）を拾わない
    excludes=("ニュー", "new", "ntlg", "わんぱく", "wanpaku", "マリア", "maria", "カトレア", "cattleya",
              "ハミング", "humming", "ポップ", "pop", "エクストラ", "rounded"),
)
TSUKUSHI_MIDASHI = FamilySpec(("筑紫a見出ミン", "tsukushiamidashimin", "tsukuamidashimin"))
KAIMIN_SORA = FamilySpec(("解ミン宙", "kaiminsora"))

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

# どれかの組が探す書体（_installed_faces は、まずこれに名前が合うファミリーだけに絞る）
_ALL_SPECS = tuple(dict.fromkeys(
    spec for preset in PRESETS.values() for spec in (*preset.heading, *preset.numeral, *preset.text)
))

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


def _name_matches(spec: FamilySpec, name: str) -> bool:
    """``_normalize`` 済みの名前 ``name`` が ``spec`` の書体か。"""
    return any(key in name for key in spec.keys) and not any(ex in name for ex in spec.excludes)


def matches(spec: FamilySpec, family: str) -> bool:
    return _name_matches(spec, _normalize(family))


def _wanted(family: str) -> bool:
    """どれかの組が探す書体か（名前だけで決める）。"""
    name = _normalize(family)
    return any(_name_matches(spec, name) for spec in _ALL_SPECS)


def pick_face(specs: tuple[FamilySpec, ...], faces: Sequence[Face], weight: int) -> Face | None:
    """``specs`` の順に探し、見つかった書体のうち ``weight`` に一番近い太さのものを返す。

    同じ近さなら太い方（見出しや数字が細る方向に外れないように）。
    """
    for spec in specs:
        hits = [face for face in faces if matches(spec, face.family)]
        if hits:
            return min(hits, key=lambda f: (abs(f.weight - weight), -f.weight))
    return None


# ---------------------------------------------------------------------------
# 書体の組（窓が 1 つ持つ）
# ---------------------------------------------------------------------------


def preset_name(name: str | None) -> str:
    """設定 ``GAUGE_FONT_PRESET`` の値を組の名前にそろえる。空なら既定、知らない名前は知らせて既定にする。"""
    name = (name or "").strip().lower() or DEFAULT_PRESET
    if name not in PRESETS:
        print(f"[ゲージ] 知らない書体の組: {name!r}。{DEFAULT_PRESET!r} を使います"
              f"（選べるもの: {', '.join(PRESETS)}）", file=sys.stderr)
        name = DEFAULT_PRESET
    return name


class FontSet:
    """1 つの書体の組。役・大きさ・太さごとの ``QFont`` と、文字列の幅を覚えておく。

    ゲージの窓（``GaugeWidget``）が 1 つ持って描くときに引く。組を変えるときは中身を書き換えず、別の
    ``FontSet`` に差し替える（窓は ``FontSet`` の同一性で、動かない層を作り直すかを決める）。
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = preset_name(name)
        self.preset = PRESETS[self.name]
        self._faces: dict[tuple[str, int], Face | None] = {}
        self._fonts: dict[tuple[str, int, int], object] = {}
        # 文字列の幅。値の文字は 30 Hz で変わるので、覚える数に上限を置く
        self._advance = functools.lru_cache(maxsize=1024)(self._measure)

    def __repr__(self) -> str:
        return f"FontSet({self.name!r})"

    def resolve(self, role: str, weight: int) -> tuple[Face | None, tuple[str, ...]]:
        """役と太さに合う書体（無ければ None）と、見つからないときの代わりの並び。

        探すのは（役・太さ）ごとに 1 度だけ（大きさが違っても同じ書体）。
        """
        key = (role, weight)
        if key not in self._faces:
            specs: tuple[FamilySpec, ...] = getattr(self.preset, role)
            self._faces[key] = pick_face(specs, _installed_faces(), weight) if specs else None
        fallback = (self.preset.heading_fallback if role == "heading" else ()) + _gothic_fallback()
        return self._faces[key], fallback

    def font(self, role: str, pixel_size: int, weight: int):
        """``QFont``。同じ（役・大きさ・太さ）は使い回す（毎フレーム作り直さない）。

        返すのは覚えておいた ``QFont`` そのもの。呼ぶ側は書き換えない（``QPainter.setFont`` は
        複製して持つので、そのまま渡してよい）。
        """
        key = (role, pixel_size, weight)
        font = self._fonts.get(key)
        if font is None:
            font = self._fonts[key] = self._make_font(role, pixel_size, weight)
        return font

    def text_advance(self, text: str, role: str, pixel_size: int, weight: int) -> float:
        """文字列の幅（``QFontMetricsF.horizontalAdvance``）。同じものは測り直さない。"""
        return self._advance(text, role, pixel_size, weight)

    def _measure(self, text: str, role: str, pixel_size: int, weight: int) -> float:
        from app.core.qt import QtGui

        return QtGui.QFontMetricsF(self.font(role, pixel_size, weight)).horizontalAdvance(text)

    def _make_font(self, role: str, pixel_size: int, weight: int):
        from app.core.qt import QtGui

        face, fallback = self.resolve(role, weight)
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
        return font


@functools.cache
def _shared_font_set(name: str) -> FontSet:
    return FontSet(name)


def font_set(name: str | None = None) -> FontSet:
    """組の名前（設定 ``GAUGE_FONT_PRESET`` の値。空・知らない名前は既定）の ``FontSet``。同じ組は使い回す。"""
    return _shared_font_set(preset_name(name))


# ---------------------------------------------------------------------------
# Qt 側
# ---------------------------------------------------------------------------


@functools.cache
def _installed_faces() -> tuple[Face, ...]:
    """この Mac の書体のうち、どれかの組が探すもの（family × style）。初回だけ引く。

    先にファミリー名だけで絞り、合うものだけ styles・weight を問い合わせる（書体を大量に入れた Mac では、
    全ファミリーに問い合わせると初回の描画が 100 ms〜1 s 止まる）。
    """
    from app.core.qt import QtGui

    db = QtGui.QFontDatabase
    faces = []
    for family in db.families():
        if not _wanted(family):
            continue
        for style in db.styles(family) or [""]:
            weight = _weight_from_name(family, style)
            if weight is None:
                weight = int(db.weight(family, style))  # Qt 6 の QFont.Weight は CSS と同じ 100〜900 の尺度
            faces.append(Face(family, style, weight))
    return tuple(faces)


@functools.cache
def _gothic_fallback() -> tuple[str, ...]:
    """ヒラギノ角ゴ → 同梱の IPAex ゴシック（初回だけ登録する）。"""
    from app.core.qt import QtGui
    from app.core.resources import japanese_font_path

    families = ["Hiragino Sans", "Hiragino Kaku Gothic ProN"]
    font_id = QtGui.QFontDatabase.addApplicationFont(str(japanese_font_path()))
    if font_id != -1:
        families.extend(QtGui.QFontDatabase.applicationFontFamilies(font_id))
    return tuple(families)


def _enable_tabular_figures(font) -> None:
    """等幅数字（OpenType の tnum）。QFont.setFeature は Qt 6.7 から。無ければ何もしない。"""
    from app.core.qt import QtGui

    try:
        font.setFeature(QtGui.QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# 診断: python -m app.gauge.fonts [組の名前]
# ---------------------------------------------------------------------------


def _report(selected: str) -> str:
    """組ごとに、見つかった書体（無ければ代わりの並び）と、この Mac のフォントワークスの書体を一覧する。"""
    from app.core.qt import QtGui

    lines = []
    for name, preset in PRESETS.items():
        mark = "（選んだ組）" if name == selected else ""
        lines.append(f"■ {name}: {preset.label}{mark}")
        fonts = FontSet(name)
        for role, weight in (("heading", 800), ("numeral", 800), ("text", 700)):
            face, fallback = fonts.resolve(role, weight)
            if face is not None:
                found = f"{face.family} / {face.style or '-'}（太さ {face.weight}）"
            elif getattr(preset, role):
                found = f"見つからない → {' → '.join(fallback)}"
            else:
                found = "ヒラギノ角ゴ → IPAex（従来）"
            lines.append(f"    {role:<8} {found}")
    fontworks = sorted(
        family for family in QtGui.QFontDatabase.families()
        if _normalize(family).startswith("fot") or _wanted(family)
    )
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
    # 組の名前を渡すと、その組に印を付ける（省略すると既定の組）
    print(_report(preset_name(args[0] if args else None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
