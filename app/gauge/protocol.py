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

import codecs
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
    "LineDemux",
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


def finite_or_none(value: Any) -> float | None:
    """有限の数なら float に、それ以外（None・NaN・無限・数でないもの）なら None にする。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


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
    """``[lo, hi]`` で ``lo < hi`` かつ ``hi > 0`` のものだけを帯にする。

    上端は弧の割合 ``now / (1.25·hi)``（``model.fraction``・``scene.build_scene``）の分母になるので、
    0 以下を通すと場面の組み立てが ZeroDivisionError で落ちる（正の仕事の帯なので正しい送り主は出さない）。
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    lo, hi = raw[0], raw[1]
    if not (_is_number(lo) and _is_number(hi)):
        return None
    lo_f, hi_f = float(lo), float(hi)
    if not (lo_f < hi_f and hi_f > 0):
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


class LineDemux:
    """子プロセスの出力の塊（bytes）から、ゲージの行だけを拾い出す。

    ``QProcess`` の ``MergedChannels``（GUI 側。このクラスは呼ばない）は
    ``readAllStandardOutput()`` の bytes を塊単位で渡してくる。行の途中で
    塊が切れることは普通にある（パイプのバッファ次第）ので、行として組み直す
    役目をここに切り出す。子プロセス側の解析スクリプトが ``\\r`` で進捗表示を
    書き換えることがあり、それを改行が来るまで止めてしまうと使い勝手が悪いので、
    「ゲージの行らしい途中」だけをため、それ以外の普通の行はすぐログへ流す。

    Qt には依存しない（``app.gauge.protocol`` 全体の方針と同じ）。呼び出し側
    （後で足す ``app/runners/worker.py``）が Qt のシグナルに変換する。
    """

    def __init__(self) -> None:
        # UTF-8 の増分デコーダ。マルチバイト文字が塊の境目で切れても、
        # 完成するまで内部でためてくれる（手で bytes をためる必要がない）。
        # errors="replace" は、本当に壊れたバイト列（子プロセスのバグ等）を
        # 例外にせず U+FFFD に変えて先へ進むため。
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # 改行でまだ終わっていない、今組み立て中の行（文字列）。
        self._pending = ""

    def feed(self, data: bytes) -> tuple[str, list[GaugeFrame]]:
        """塊を1つ渡す。戻り値は (ログへ出す文字列, 拾えたフレームの列)。"""
        self._pending += self._decoder.decode(data)

        log_parts: list[str] = []
        frames: list[GaugeFrame] = []

        # 行ごとに残りを切り直すと塊が大きいとき O(k²) になるので、
        # 開始位置だけ進めて、最後に1回だけ切り詰める。
        pending = self._pending
        start = 0
        while True:
            newline_at = pending.find("\n", start)
            if newline_at == -1:
                break
            self._consume_line(pending[start : newline_at + 1], log_parts, frames)
            start = newline_at + 1
        self._pending = pending[start:]

        # 残り（改行なしの途中の行）。ゲージの行になりうる途中だけをため、
        # それ以外はここで確定させてすぐログへ流す（\r の進捗表示を止めないため）。
        if self._pending and not self._looks_like_gauge_prefix(self._pending):
            prefix_at = self._pending.find(PREFIX)
            if prefix_at == -1:
                # 行の途中の PREFIX が塊の境目で割れた場合（"…@@GA" ＋ "UGE {…}"）。
                # 末尾の PREFIX の頭になりうる部分だけをため、手前はログへ流す。
                keep = self._partial_prefix_len(self._pending)
                log_parts.append(self._pending[: len(self._pending) - keep])
                self._pending = self._pending[len(self._pending) - keep :]
            else:
                # 行の途中に PREFIX が現れた場合（任意の要件）。手前はログへ、
                # PREFIX から先はゲージの行の候補としてためておく。
                log_parts.append(self._pending[:prefix_at])
                self._pending = self._pending[prefix_at:]

        return "".join(log_parts), frames

    def flush(self) -> str:
        """ためている残り（改行が来ないまま終わった分）をログとして返す。"""
        # デコーダの内部にも、まだ完成していないマルチバイト列の断片が
        # 残っていることがある（プロセスが行の途中・文字の途中で終了した場合）。
        # final=True で確定させ、壊れていれば errors="replace" で置き換える。
        tail = self._pending + self._decoder.decode(b"", final=True)
        self._pending = ""
        return tail

    def reset(self) -> None:
        """バッファとデコーダを初期化する（前回の計測の続きと混ざらないように）。"""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""

    @staticmethod
    def _partial_prefix_len(text: str) -> int:
        """``text`` の末尾が PREFIX の頭（全体ではない）と一致する最長の長さ。無ければ 0。"""
        for k in range(min(len(PREFIX) - 1, len(text)), 0, -1):
            if text.endswith(PREFIX[:k]):
                return k
        return 0

    @staticmethod
    def _looks_like_gauge_prefix(text: str) -> bool:
        """``text`` が PREFIX の頭の一部、または PREFIX で始まる途中か。"""
        return PREFIX.startswith(text) or text.startswith(PREFIX)

    @classmethod
    def _consume_line(cls, line: str, log_parts: list[str], frames: list[GaugeFrame]) -> None:
        """改行で終わった1行を、フレームかログへ振り分ける。"""
        if line.startswith(PREFIX):
            cls._consume_gauge_candidate(line, log_parts, frames)
            return
        # 行の途中に PREFIX が現れた場合（任意の要件）。手前はログへ、
        # PREFIX から先だけをゲージの行の候補として decode する。
        prefix_at = line.find(PREFIX)
        if prefix_at > 0:
            log_parts.append(line[:prefix_at])
            cls._consume_gauge_candidate(line[prefix_at:], log_parts, frames)
            return
        log_parts.append(line)

    @staticmethod
    def _consume_gauge_candidate(line: str, log_parts: list[str], frames: list[GaugeFrame]) -> None:
        """``PREFIX`` で始まり改行で終わる1行を decode し、壊れていればログへ。"""
        frame = decode(line)
        if frame is not None:
            frames.append(frame)
        else:
            log_parts.append(line)
