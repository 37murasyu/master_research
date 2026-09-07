"""スマホ ⇄ PC のメッセージ定義。Android 側と PC 側の**契約**。

設計の要点は「遅延」と「同期ずれ」を分けて扱うこと。

三角測量を壊すのは 2 台のカメラの**相対的なずれ**であって、両方が揃って
遅れること自体は害がない（ゲージの表示が遅れるだけ）。したがって

1. 各フレームに撮影時刻を刻んで送る
2. 受信側は時刻でペアリングする（到着順では組まない）

とすれば、1 秒程度のラグを許容できる。時刻合わせは NTP と同じ往復測定で行い、
**PC 自身を時刻サーバにする**。両端末が同一の PC 時計に揃うので、結果として
端末どうしも揃う。

この発想はこのリポジトリに前例がある。HX711 ロードセルは
``master_research_code.py:2338, 2365-2368`` で ``hx_start_iso`` を送って
開始時刻を合わせている。映像側にだけ同等の機構が無かった。

ネットワーク越しの入力は信用できないので、``decode`` は必ず検証してから返す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "PROTOCOL_VERSION",
    "PixelCoordinates",
    "LANDMARK_COUNT",
    "ROLES",
    "ProtocolError",
    "LandmarkFrame",
    "SyncRequest",
    "SyncResponse",
    "Hello",
    "ClockOffset",
    "encode",
    "decode",
    "compute_clock_offset",
    "best_offset",
]

PROTOCOL_VERSION = 1

# MediaPipe Pose のランドマーク数。Tasks API の pose_landmarker_lite.task が
# 返す点数と一致する。PC 側が実際に使うのは config.pose_keypoints の 12 点だが、
# 帯域が無視できる（33点でも約 600 B/frame）ので全点送って将来の余地を残す。
LANDMARK_COUNT = 33

# ステレオの左右。既存コードの cam0 / cam1 に対応する。
ROLES = ("cam0", "cam1")


class ProtocolError(ValueError):
    """受信したメッセージが契約を満たさない。

    JSON の破損・型違い・値域外をすべてこれに集約するので、
    呼び出し側は 1 種類だけ捕まえればよい。
    """


# ---------------------------------------------------------------------------
# メッセージ
# ---------------------------------------------------------------------------
class PixelCoordinates:
    """``landmarks`` / ``width`` / ``height`` を持つ型に、ピクセル換算を与える。

    「正規化座標に w/h を掛けてピクセルにする」という規約はプロトコルの一部で、
    ``tests/test_protocol_contract.py`` が検証している。送信側の
    ``LandmarkFrame`` と受信側の ``sync_buffer.InterpolatedFrame`` が
    別々に実装していると、片方だけ直したときに黙ってずれる。

    フィールドを持たないミックスインにしてあるのは、両者の
    コンストラクタの形（seq の有無、時刻フィールドの名前）を変えないため。
    """

    landmarks: Sequence[tuple[float, float, float, float]]
    width: int
    height: int

    def pixel_xy(self, index: int) -> tuple[float, float]:
        """正規化座標をピクセル座標に直す。

        既存 ``utils.extract_keypoints`` が
        ``landmark.x * frame.shape[1]`` としているのと同じ規約に揃える。
        """
        x, y, _z, _v = self.landmarks[index]
        return (x * self.width, y * self.height)


@dataclass(frozen=True)
class LandmarkFrame(PixelCoordinates):
    """1 フレーム分の姿勢ランドマーク。"""

    role: str
    seq: int
    # PC の時計に補正済みの撮影時刻。端末のローカル時計ではない。
    t_capture_ns: int
    width: int
    height: int
    # (x, y, z, visibility)。x, y は [0,1] の正規化座標。
    landmarks: Sequence[tuple[float, float, float, float]]


@dataclass(frozen=True)
class SyncRequest:
    """端末 → PC。``t1`` は端末の単調時計（Android の elapsedRealtimeNanos）。"""

    t1: int


@dataclass(frozen=True)
class SyncResponse:
    """PC → 端末。``t2`` は受信時刻、``t3`` は送信時刻（どちらも PC 時計）。"""

    t1: int
    t2: int
    t3: int


@dataclass(frozen=True)
class Hello:
    """接続時の名乗り。role で cam0 / cam1 を決める。"""

    role: str
    device: str
    session: str


Message = LandmarkFrame | SyncRequest | SyncResponse | Hello


# ---------------------------------------------------------------------------
# 符号化・復号
# ---------------------------------------------------------------------------
def encode(message: Message) -> str:
    if isinstance(message, LandmarkFrame):
        payload: dict[str, Any] = {
            "type": "landmarks",
            "role": message.role,
            "seq": message.seq,
            "t_capture_ns": message.t_capture_ns,
            "w": message.width,
            "h": message.height,
            "lm": [list(point) for point in message.landmarks],
        }
    elif isinstance(message, SyncRequest):
        payload = {"type": "sync_req", "t1": message.t1}
    elif isinstance(message, SyncResponse):
        payload = {
            "type": "sync_res",
            "t1": message.t1,
            "t2": message.t2,
            "t3": message.t3,
        }
    elif isinstance(message, Hello):
        payload = {
            "type": "hello",
            "v": PROTOCOL_VERSION,
            "role": message.role,
            "device": message.device,
            "session": message.session,
        }
    else:  # pragma: no cover - 型で塞いである
        raise TypeError(f"未知のメッセージ型: {type(message)!r}")

    return json.dumps(payload, separators=(",", ":"))


def decode(raw: str | bytes) -> Message:
    """受信文字列をメッセージに変換する。契約違反は ProtocolError。"""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"JSON として読めません: {exc}") from exc

    if not isinstance(payload, dict):
        raise ProtocolError(f"オブジェクトではありません: {type(payload).__name__}")

    kind = payload.get("type")
    if kind == "landmarks":
        return _decode_landmarks(payload)
    if kind == "sync_req":
        return SyncRequest(t1=_require_int(payload, "t1"))
    if kind == "sync_res":
        return SyncResponse(
            t1=_require_int(payload, "t1"),
            t2=_require_int(payload, "t2"),
            t3=_require_int(payload, "t3"),
        )
    if kind == "hello":
        return Hello(
            role=_require_role(payload),
            device=_require_str(payload, "device"),
            session=_require_str(payload, "session"),
        )

    raise ProtocolError(f"未知のメッセージ種別: {kind!r}")


def _decode_landmarks(payload: dict[str, Any]) -> LandmarkFrame:
    width = _require_int(payload, "w")
    height = _require_int(payload, "h")
    if width <= 0 or height <= 0:
        raise ProtocolError(f"フレームサイズが不正: w={width}, h={height}")

    raw_points = payload.get("lm")
    if not isinstance(raw_points, list):
        raise ProtocolError("lm が配列ではありません")
    if len(raw_points) != LANDMARK_COUNT:
        raise ProtocolError(
            f"ランドマーク数が {len(raw_points)}。{LANDMARK_COUNT} 点である必要があります"
        )

    points: list[tuple[float, float, float, float]] = []
    for i, point in enumerate(raw_points):
        if not isinstance(point, (list, tuple)) or len(point) != 4:
            raise ProtocolError(f"lm[{i}] は (x, y, z, visibility) の4要素である必要があります")
        try:
            points.append((float(point[0]), float(point[1]), float(point[2]), float(point[3])))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"lm[{i}] に数値でない値: {point!r}") from exc

    return LandmarkFrame(
        role=_require_role(payload),
        seq=_require_int(payload, "seq"),
        t_capture_ns=_require_int(payload, "t_capture_ns"),
        width=width,
        height=height,
        landmarks=points,
    )


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    # bool は int の派生なので明示的に弾く（True が 1 として通ると気づきにくい）
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} は整数である必要があります: {value!r}")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ProtocolError(f"{key} は文字列である必要があります: {value!r}")
    return value


def _require_role(payload: dict[str, Any]) -> str:
    role = payload.get("role")
    if role not in ROLES:
        raise ProtocolError(f"role が不正: {role!r}（有効な値: {', '.join(ROLES)}）")
    return role


# ---------------------------------------------------------------------------
# 時刻同期
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockOffset:
    """1 回の往復測定の結果。

    offset_ns: 端末時計に足すと PC 時計になる差分
    rtt_ns:    往復遅延（サーバ内の処理時間を除く）
    """

    offset_ns: int
    rtt_ns: int


def compute_clock_offset(t1: int, t2: int, t3: int, t4: int) -> ClockOffset:
    """NTP と同じ式で時計ずれと往復遅延を求める。

    t1: 端末が送信した時刻（端末時計）
    t2: PC が受信した時刻（PC 時計）
    t3: PC が返信した時刻（PC 時計）
    t4: 端末が受信した時刻（端末時計）

    往路と復路の遅延が等しいと仮定して、ずれを片道分ずつ打ち消す。
    RTT からサーバ内の処理時間 (t3-t2) を除くのは、その間は
    ネットワークを飛んでいないため。
    """
    offset = ((t2 - t1) + (t3 - t4)) // 2
    rtt = (t4 - t1) - (t3 - t2)
    return ClockOffset(offset_ns=offset, rtt_ns=rtt)


def best_offset(samples: Iterable[ClockOffset]) -> ClockOffset | None:
    """複数回の測定から最も信頼できるものを選ぶ。

    RTT が最小のサンプルを採る。往復が速かった回ほど、
    「往路と復路が等しい」という仮定からのずれが小さいため。
    平均を取らないのは、外れ値（一時的な輻輳）に引きずられるのを避けるため。
    """
    samples = list(samples)
    if not samples:
        return None
    return min(samples, key=lambda s: s.rtt_ns)
