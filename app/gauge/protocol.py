"""被験者ゲージの行の書式 v2（``@@GAUGE ``）。計測の子プロセスと GUI の契約。

計測の子プロセス（別セッションが実装）は毎フレーム
``sys.stdout.write(encode(frame))`` で 1 行を出す。GUI（Qt）側は
``QProcess`` の標準出力を行単位に区切って ``decode`` に渡す。

子プロセスは Qt を読まない（Qt を持たない環境で計測だけ回すことがある）ので、
このモジュールは標準ライブラリだけで書く。``app.gauge`` パッケージの他のモジュール
（GUI 側の描画）から独立して import できることを、
``test_protocol_does_not_import_qt`` が別プロセスで確かめている。

子プロセスからの入力はプロセス境界を越えた文字列であり信用できない。
``decode`` は ``app.net.protocol`` の ``ProtocolError`` 方式とは違い、
例外を投げず ``None`` を返す設計にしてある。GUI の描画ループは 30Hz で
1 行ずつ読み続けるので、壊れた 1 行のたびに例外処理を挟むより
「None なら前回の表示を保つ」で済ませたほうが単純に保てる。

行の形式（controller の constraints.md と合意済み。値・鍵の名前は変えない）::

    @@GAUGE {"v":2,"link":"waiting|connected","rep":N,"source":"measure|demo|replay",
             "parts":{"elbow_L":{"now":12.3,"prev":10.1,"band":[lo,hi],"w1rm":66.0}, …}}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "PREFIX",
    "VERSION",
    "PART_NAMES",
    "LINKS",
    "SOURCES",
    "PartReading",
    "GaugeFrame",
    "encode",
    "decode",
]

# 1 行の接頭辞。GUI 側はこれで「ゲージの行」と、子プロセスが誤って（あるいは
# ライブラリが）標準出力に出してしまう他の文字列とを見分ける。
PREFIX = "@@GAUGE "

VERSION = 2

# GUI が描く 4 部位。順序は表示順でもある。
PART_NAMES = ("elbow_L", "elbow_R", "wrist_L", "wrist_R")

LINKS = ("waiting", "connected")

SOURCES = ("measure", "demo", "replay")


@dataclass(frozen=True)
class PartReading:
    """1 部位・1 フレーム分の読み。

    ``now`` が None なのは「まだこの部位の値が出ていない」（接続待ちや、
    その部位を計測していない瞬間）。``prev`` が None なのは「まだ 1 回も
    完了していない」（1 レップ目の途中）。どちらも「値が無い」を意味するが、
    別の理由なので別のフィールドにしてある。
    """

    now: float | None
    prev: float | None = None
    band: tuple[float, float] | None = None
    w1rm: float | None = None


@dataclass(frozen=True)
class GaugeFrame:
    """1 行分（＝1 フレーム分）の被験者ゲージの状態。"""

    link: str
    rep: int
    source: str = "measure"
    parts: Mapping[str, PartReading] = field(default_factory=dict)


def _encode_number(value: float | None) -> float | None:
    """数は 0.1 に丸め、NaN・±inf は null にする。

    非有限をそのまま JSON へ出すと（``json.dumps`` は既定で NaN/Infinity という
    非標準の字句を書いてしまい）、GUI 側でない別のパーサが読めなくなる。
    """
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(value, 1)


def _encode_band(band: tuple[float, float] | None) -> list[float] | None:
    """band は [lo, hi] の 2 要素。片方でも非有限なら組として無効なので null にする。"""
    if band is None:
        return None
    lo, hi = band
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return [round(lo, 1), round(hi, 1)]


def encode(frame: GaugeFrame) -> str:
    """``PREFIX`` + JSON（区切り詰め・ASCII・1 行 512 バイト未満）+ ``"\\n"``。"""
    parts_payload: dict[str, dict[str, Any]] = {}
    for name, reading in frame.parts.items():
        parts_payload[name] = {
            "now": _encode_number(reading.now),
            "prev": _encode_number(reading.prev),
            "band": _encode_band(reading.band),
            "w1rm": _encode_number(reading.w1rm),
        }

    payload = {
        "v": VERSION,
        "link": frame.link,
        "rep": frame.rep,
        "source": frame.source,
        "parts": parts_payload,
    }
    # 区切りを詰めるのと ASCII 化は、どちらも 1 行を 512 バイト（macOS の PIPE_BUF）
    # 未満に収めるため。ensure_ascii は既定で True だが意図を明示しておく。
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return f"{PREFIX}{body}\n"


def _is_number(value: Any) -> bool:
    """「数か null」の「数」。bool と非有限（NaN・±inf）は数として扱わない。

    bool は int の派生なので明示的に弾く（True が 1 として通ると気づきにくい）。
    契約では「NaN は null で来る」が、これは正しい送り主の振る舞いであって
    decode 側の保証ではない。壊れた・悪意ある送り主が生の NaN／Infinity を
    JSON リテラルとして送ってきても（``json.loads`` は既定でこれを受理する）、
    isinstance だけでは通ってしまうので、ここで isfinite も確かめる。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _decode_band(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    lo, hi = raw[0], raw[1]
    if not (_is_number(lo) and _is_number(hi)):
        return None
    lo_f, hi_f = float(lo), float(hi)
    if not lo_f < hi_f:
        return None
    return (lo_f, hi_f)


def _decode_part(raw: Any) -> PartReading | None:
    if not isinstance(raw, dict):
        return None
    now = raw.get("now")
    if now is not None and not _is_number(now):
        # now が数でも null でもない部位は、丸ごと捨てる（GUI 側は「値なし」と区別できない
        # 壊れ方なので、中途半端な PartReading を作らない）
        return None
    prev = raw.get("prev")
    w1rm = raw.get("w1rm")
    return PartReading(
        now=float(now) if now is not None else None,
        prev=float(prev) if _is_number(prev) else None,
        band=_decode_band(raw.get("band")),
        w1rm=float(w1rm) if _is_number(w1rm) else None,
    )


def decode(line: str) -> GaugeFrame | None:
    """1 行を ``GaugeFrame`` に直す。契約を満たさなければ ``None``。

    子プロセスからの入力は信用できないので、ここで弾けるものはすべて弾く。
    呼び出し側（GUI の読み取りループ）は None かどうかだけ見ればよい。
    """
    if not line.startswith(PREFIX):
        return None
    # 末尾の改行・CR を落とす。QProcess の MergedChannels は \n 区切りで届くが、
    # 送り主が \r\n を書いてくることも受け付ける。
    body = line[len(PREFIX) :].rstrip("\r\n")

    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    version = payload.get("v")
    # 型も一致させる（bool は type() が int にならないのでここで弾ける）。
    # ``version == VERSION`` だけだと、浮動小数の 2.0 が Python の等価規則で
    # 通ってしまう（2.0 == 2 は真）。文字列の "2" と同様に、数値でも整数
    # そのもの以外は版違いとして扱う。
    if type(version) is not int or version != VERSION:
        return None

    link = payload.get("link")
    if link not in LINKS:
        return None

    rep = payload.get("rep")
    if isinstance(rep, bool) or not isinstance(rep, int) or rep < 0:
        return None

    source = payload.get("source")
    if source not in SOURCES:
        return None

    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, dict):
        return None

    parts: dict[str, PartReading] = {}
    for name in PART_NAMES:
        if name not in raw_parts:
            continue
        reading = _decode_part(raw_parts[name])
        if reading is not None:
            parts[name] = reading

    return GaugeFrame(link=link, rep=rep, source=source, parts=parts)
