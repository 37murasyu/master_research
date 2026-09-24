"""スマホとの接続を背景スレッドで回し、メインスレッドと受け渡す。

混成構成のランナーは 1 プロセスに 2 本のスレッドを持つ。

- **メインスレッド**: PC のカメラの取り込み、MediaPipe、``cv.imshow``（macOS では
  メインスレッドでしか窓を扱えない）、停止の判定。
- **ループのスレッド**（ここ）: asyncio の ``LandmarkServer``。スマホの受信、時刻同期、
  ペアリング、撮影要求。

同期バッファにはロックが無いので、PC 側の点もループのスレッドへ渡してから入れる
（``inject``）。メインスレッドへ返すもの（最新の画像、直近の点、状態）はロックで守る。

``on_pairs`` と ``on_landmarks`` は**ループのスレッドで**呼ばれる。中で例外が起きても
受信は止めない（数えて、最初の 1 回だけ内容を出す）。
"""

from __future__ import annotations

import asyncio
import bisect
import collections
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Iterable

from app.net import protocol as p
from app.net.server import DEFAULT_PORT, LandmarkServer, check_injectable, local_ip
from app.net.sync_buffer import GridSpec, PairedSample, SyncBuffer

__all__ = [
    "CALIBRATION",
    "CaptureMode",
    "CaptureScheduler",
    "LinkStatus",
    "OFF",
    "PREVIEW",
    "PhoneLink",
]


# ---------------------------------------------------------------------------
# 撮影要求の出し方
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureMode:
    """撮影要求の頻度と、返してもらう画像の大きさ・画質。"""

    hz: float
    max_width: int | None = None
    quality: int | None = None


# 撮影要求を出さない。
OFF = CaptureMode(hz=0.0)
# 向き合わせ用のライブ表示。640 px・画質 70 なら全解像度の 1/5 程度の大きさで済む。
PREVIEW = CaptureMode(hz=4.0, max_width=640, quality=70)
# 校正用。全解像度・画質 90。盤の角点を画素未満の精度で拾うため縮めない。
CALIBRATION = CaptureMode(hz=3.0, quality=90)


class CaptureScheduler:
    """撮影要求をいつ出すかを決める。時計を差し込める純粋なロジック。

    - 同時に出すのは 1 件だけ。端末の JPEG 化が追いつかないまま要求を溜めない
    - 前回を出してから ``1/hz`` 秒たつまで次を出さない
    - 応答が ``timeout_s`` 秒来なければ諦めて次を出す。電文が消えた、端末が撮影に
      対応していない（古いアプリ）、などで止まらないように
    """

    def __init__(
        self,
        mode: CaptureMode = OFF,
        timeout_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._mode = mode
        self._timeout_s = timeout_s
        self._clock = clock
        self._next_id = 1
        self._outstanding: int | None = None
        self._issued_at: float | None = None
        self.timeouts = 0
        self._revision = 0
        self._request_revisions: dict[int, int] = {}

    @property
    def mode(self) -> CaptureMode:
        return self._mode

    def set_mode(self, mode: CaptureMode) -> None:
        if mode != self._mode:
            self._revision += 1
            self._outstanding = None
            self._issued_at = None
        self._mode = mode

    def accepts_capture(self, capture_id: int) -> bool:
        """切替前の縮小 JPEG を全解像度の校正画像として使わない。"""
        return self._request_revisions.get(capture_id) == self._revision

    def next_request(self) -> p.CaptureRequest | None:
        """今出すべき要求があれば作って返す（出したものとして記録する）。"""
        if self._mode.hz <= 0:
            return None
        now = self._clock()
        if self._outstanding is not None:
            assert self._issued_at is not None
            if now - self._issued_at < self._timeout_s:
                return None
            self.timeouts += 1
            self._outstanding = None
        if self._issued_at is not None and now - self._issued_at < 1.0 / self._mode.hz:
            return None

        request = p.CaptureRequest(
            id=self._next_id, max_width=self._mode.max_width, quality=self._mode.quality
        )
        self._request_revisions[request.id] = self._revision
        if len(self._request_revisions) > 64:
            del self._request_revisions[next(iter(self._request_revisions))]
        self._next_id += 1
        self._outstanding = request.id
        self._issued_at = now
        return request

    def complete(self, capture_id: int) -> bool:
        """応答が来た。待っていた要求なら解除して True。"""
        if capture_id != self._outstanding:
            return False
        self._outstanding = None
        return True


# ---------------------------------------------------------------------------
# 状態
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkStatus:
    """メインスレッドへ渡す状態の写し。ループのスレッドが定期的に作り直す。"""

    url: str = ""
    devices: dict[str, p.Hello] = field(default_factory=dict)
    clients: int = 0
    frames_received: int = 0
    frames_injected: int = 0
    # 直近 1 秒に届いた端末の点の数（撮影時刻で数える）。
    remote_fps: float = 0.0
    pairs: int = 0
    dropped_gap: int = 0
    dropped_late: int = 0
    mean_skew_ms: float = 0.0
    max_skew_ms: float = 0.0
    protocol_errors: int = 0
    role_mismatches: int = 0
    captures_received: int = 0
    capture_timeouts: int = 0
    callback_errors: int = 0


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
PairsCallback = Callable[[Iterable[PairedSample]], None]
LandmarksCallback = Callable[[p.LandmarkFrame], None]
HelloCheck = Callable[[p.Hello], "str | None"]


class PhoneLink:
    """スマホ 1 台（``remote_role``）を受け、PC のカメラの点を ``inject`` で混ぜる。"""

    def __init__(
        self,
        *,
        remote_role: str = "cam1",
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        advertise_host: str | None = None,
        target_hz: float = 30.0,
        window_sec: float = 2.0,
        max_gap_ms: float = 100.0,
        grid: GridSpec | None = None,
        on_pairs: PairsCallback | None = None,
        on_landmarks: LandmarksCallback | None = None,
        on_hello: HelloCheck | None = None,
        capture_mode: CaptureMode = OFF,
        capture_timeout_s: float = 1.0,
        history: int = 90,
        accept_frame: Callable[[p.LandmarkFrame], bool] | None = None,
        on_tick: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        session: str | None = None,
    ):
        """``session`` を省略すると起動ごとに作る。ランナーは ``stable_session()`` を渡し、
        Pixel が覚えた接続先へ自動でつなぎ直せるようにする。

        ``grid``（``GridSpec``）を渡すと ``target_hz``・``max_gap_ms`` より優先する。計測のランナーは
        ``MeasurementConfig.grid`` を渡し、同期バッファと計測の格子を揃える。"""
        self.remote_role = remote_role
        self._user_on_tick = on_tick
        self._user_on_stop = on_stop
        self._advertise_host = advertise_host
        self._user_on_pairs = on_pairs
        self._user_on_landmarks = on_landmarks
        self._user_on_hello = on_hello

        # LandmarkServer の生成はループを必要としないので、ここで作る（session と
        # URL をすぐ使えるように）。起動はループのスレッドで行う。
        self._server = LandmarkServer(
            host=host,
            port=port,
            buffer=SyncBuffer(target_hz=target_hz, window_sec=window_sec, max_gap_ms=max_gap_ms, grid=grid),
            session=session,
            on_pairs=self._deliver_pairs,
            on_landmarks=self._handle_landmarks,
            on_calibration_frame=self._handle_capture,
            on_hello=self._check_hello,
            remote_roles=(remote_role,),
            accept_frame=accept_frame,
            require_hello=True,
            # GUI からの停止は 2 秒以内に終える。Pixel が応答しない（画面を消した、
            # Wi-Fi が切れた）ときに、閉じる挨拶を既定の 10 秒も待たない。
            close_timeout=0.5,
        )
        self._scheduler = CaptureScheduler(capture_mode, timeout_s=capture_timeout_s)

        self._lock = threading.Lock()
        self._recent: collections.deque[p.LandmarkFrame] = collections.deque(maxlen=history)
        self._latest_capture: p.CalibrationFrame | None = None
        self._capture_taken = True
        self._captures_received = 0
        self._callback_errors = 0
        self._reported: set[str] = set()
        self._status = LinkStatus()

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None

    # -- 接続情報 ----------------------------------------------------------
    @property
    def session(self) -> str:
        return self._server.session

    @property
    def url(self) -> str:
        """端末に QR で渡す接続先。起動後は実際のポートが入る。"""
        host = self._advertise_host
        if host is None:
            host = local_ip() if self._server.host in ("0.0.0.0", "") else self._server.host
        return self._server.connect_url(self.remote_role, host=host)

    # -- 起動・停止 --------------------------------------------------------
    def start(self, timeout: float = 5.0) -> None:
        """サーバを起動して待ち受けが始まるまで待つ。ポートが塞がっていれば OSError。"""
        if self._thread is not None:
            raise RuntimeError("PhoneLink は既に起動しています")
        self._thread = threading.Thread(target=self._run, name="PhoneLink", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("受信サーバの起動を待てませんでした")
        if self._start_error is not None:
            self._thread.join(timeout)
            raise self._start_error

    def stop(self, timeout: float = 5.0) -> None:
        """サーバを止め、バッファに残ったペアを渡し切ってから戻る。"""
        if self._thread is None:
            return
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)
        self._thread.join(timeout)
        if self._thread.is_alive():
            print("[PhoneLink] 受信スレッドが時間内に終わりませんでした", file=sys.stderr)

    # -- メインスレッドから使うもの ----------------------------------------
    def inject(self, frame: p.LandmarkFrame) -> None:
        """PC のカメラの点を渡す。検査は呼び出し側のスレッドで行う。

        ループの中で例外になると呼び出し側に届かないため、渡す前に確かめる。
        """
        check_injectable(frame, self._server.remote_roles, self._server.buffer.roles)
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("PhoneLink が起動していません")
        loop.call_soon_threadsafe(self._server.inject, frame)

    def set_capture_mode(self, mode: CaptureMode) -> None:
        """ライブ表示・校正・停止を切り替える。"""
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._set_mode, mode)
        else:
            self._set_mode(mode)

    def _set_mode(self, mode: CaptureMode) -> None:
        if mode != self._scheduler.mode:
            with self._lock:
                self._latest_capture = None
                self._capture_taken = True
        self._scheduler.set_mode(mode)

    def take_capture(self) -> p.CalibrationFrame | None:
        """前回から新しく届いた画像があれば返す。無ければ None。"""
        with self._lock:
            if self._capture_taken:
                return None
            self._capture_taken = True
            return self._latest_capture

    def nearest_remote(self, t_ns: int, tolerance_ns: int) -> p.LandmarkFrame | None:
        """撮影時刻が ``t_ns`` に最も近い端末の点。``tolerance_ns`` より離れていれば None。"""
        with self._lock:
            frames = list(self._recent)
        if not frames:
            return None
        times = [f.t_capture_ns for f in frames]
        index = bisect.bisect_left(times, t_ns)
        candidates = [frames[i] for i in (index - 1, index) if 0 <= i < len(frames)]
        best = min(candidates, key=lambda f: abs(f.t_capture_ns - t_ns))
        return best if abs(best.t_capture_ns - t_ns) <= tolerance_ns else None

    def status(self) -> LinkStatus:
        with self._lock:
            return self._status

    # -- ループのスレッド --------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # pragma: no cover - 起動後の想定外の失敗
            if not self._ready.is_set():
                self._start_error = exc
                self._ready.set()
            else:
                traceback.print_exc()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            await self._server.start()
        except OSError as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()

        try:
            while not self._stop_event.is_set():
                await self._tick()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._server.stop()
            # 最後の数フレームがバッファに残っているので渡し切る
            remaining = self._server.buffer.drain()
            if remaining:
                self._deliver_pairs(remaining)
            self._refresh_status()
            if self._user_on_stop is not None:
                self._guarded("on_stop", self._user_on_stop)

    async def _tick(self) -> None:
        if self._user_on_tick is not None:
            self._guarded("on_tick", self._user_on_tick)
        # 端末が名乗るまでは要求を出さない（出すと、繋がる前に時間切れが積み上がる）
        if self._server.devices:
            request = self._scheduler.next_request()
            if request is not None:
                await self._server.request_capture(
                    request.id, max_width=request.max_width, quality=request.quality
                )
        self._refresh_status()

    def _refresh_status(self) -> None:
        stats = self._server.stats
        with self._lock:
            recent = [f.t_capture_ns for f in self._recent]
            captures, errors = self._captures_received, self._callback_errors
        remote_fps = 0.0
        if recent:
            newest = time.monotonic_ns()
            remote_fps = float(sum(1 for t in recent if 0 <= newest - t < 1_000_000_000))
        status = LinkStatus(
            url=self.url,
            devices=dict(self._server.devices),
            clients=int(stats["clients"]),
            frames_received=int(stats["frames_received"]),
            frames_injected=int(stats["frames_injected"]),
            remote_fps=remote_fps,
            pairs=int(stats["emitted"]),
            dropped_gap=int(stats["dropped_gap"]),
            dropped_late=int(stats["dropped_late"]),
            mean_skew_ms=float(stats["mean_role_skew_ms"]),
            max_skew_ms=float(stats["max_role_skew_ms"]),
            protocol_errors=int(stats["protocol_errors"]),
            role_mismatches=int(stats["role_mismatches"]),
            captures_received=captures,
            capture_timeouts=self._scheduler.timeouts,
            callback_errors=errors,
        )
        with self._lock:
            self._status = status

    # -- サーバからの呼び出し（ループのスレッド） --------------------------
    def _check_hello(self, hello: p.Hello) -> str | None:
        if hello.session != self._server.session:
            # 前回起動したときの QR。繋いでも、この回の校正や記録と結び付かない。
            return "古い QR です。いま表示している QR を読み直してください"
        if self._user_on_hello is not None:
            return self._user_on_hello(hello)
        return None

    def _handle_landmarks(self, frame: p.LandmarkFrame) -> None:
        if frame.role == self.remote_role:
            with self._lock:
                self._recent.append(frame)
        if self._user_on_landmarks is not None:
            self._guarded("on_landmarks", self._user_on_landmarks, frame)

    def _handle_capture(self, frame: p.CalibrationFrame) -> None:
        self._scheduler.complete(frame.id)
        if not self._scheduler.accepts_capture(frame.id):
            return
        with self._lock:
            # 時間切れの後に遅れて届いた画像も、画像としては正しいので使う
            self._latest_capture = frame
            self._capture_taken = False
            self._captures_received += 1

    def _deliver_pairs(self, pairs: Iterable[PairedSample]) -> None:
        if self._user_on_pairs is not None:
            self._guarded("on_pairs", self._user_on_pairs, pairs)

    def _guarded(self, name: str, fn: Callable, *args) -> None:
        """呼び出し先の例外で受信を止めない。数えて、最初の 1 回だけ内容を出す。"""
        try:
            fn(*args)
        except Exception:
            with self._lock:
                self._callback_errors += 1
                first = name not in self._reported
                self._reported.add(name)
            if first:
                print(f"[PhoneLink] {name} で例外が起きました（以後は数えるだけ）:", file=sys.stderr)
                traceback.print_exc()
