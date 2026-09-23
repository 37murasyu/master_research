"""記録した混成の計測（``landmarks2d_*``）を、計測中と同じ道筋で ``MeasurementSession`` に流し直す。

検証のための再生。被験者がいなくても、実機の記録や合成の押し上げを計測の全体（受け付けの検査、30 Hz 格子への補間、
三角測量、EKF、回の区切り、ゲージの値）に通して確かめられる。本番と違う道筋を通ると本番でだけ起きる不具合を
見逃すので、受信スレッド（``app.net.server``）と同じ順で呼ぶ:

    accept_frame → SyncBuffer.push → on_landmarks → SyncBuffer.drain → on_pairs

Recorder は作ったスレッドからしか書けないので、``replay`` を呼んだスレッドが記録を開いて閉じる。
GUI から使うときは環境変数 ``HYBRID_REPLAY`` に計測フォルダを入れて計測を開始する（``app.runners.hybrid_replay``）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from app.hybrid.calibration_io import load_session_calibration, update_meta
from app.hybrid.ekf import EkfSettings
from app.hybrid.measurement import MeasurementSession
from app.hybrid.retriangulate import read_landmarks
from app.net.protocol import LandmarkFrame
from app.net.sync_buffer import SyncBuffer
from app.runners.network_measure import MeasurementConfig

__all__ = ["REPLAY_ENV", "merged_frames", "replay"]

# GUI の計測（子プロセス hybrid_measure）を再生に切り替える環境変数。値は計測フォルダ
REPLAY_ENV = "HYBRID_REPLAY"
# 記録を 1 秒ごとに書き出す本番（on_tick は 50 ms ごと）に合わせた間隔
_FLUSH_INTERVAL_S = 0.05


def merged_frames(frames: Mapping[str, Sequence[LandmarkFrame]], *, start_s: float = 0.0,
                  end_s: float | None = None) -> list[LandmarkFrame]:
    """両方のカメラのフレームを撮影時刻の順に 1 本に並べ、[start_s, end_s) の窓だけを返す。

    時刻の原点は、どちらかのカメラの最初のフレーム。同じ時刻なら cam0（Mac）を先にする。
    """
    everything = [frame for role in frames.values() for frame in role]
    if not everything:
        return []
    origin = min(frame.t_capture_ns for frame in everything)
    low = origin + round(start_s * 1e9)
    high = None if end_s is None else origin + round(end_s * 1e9)
    chosen = [f for f in everything if f.t_capture_ns >= low and (high is None or f.t_capture_ns < high)]
    return sorted(chosen, key=lambda f: (f.t_capture_ns, f.role != "cam0", f.seq))


def _timing(samples_ms: list[float], pairs: int) -> dict:
    """1 組あたりの ``on_pairs`` の所要時間。受信スレッドの予算（30 Hz で 33 ms）に収まっているかを見る。"""
    if not samples_ms:
        return {"pairs": pairs, "on_pairs_ms_median": None, "on_pairs_ms_p95": None, "on_pairs_ms_max": None}
    values = np.asarray(samples_ms, dtype=float)
    return {
        "pairs": int(pairs),
        "on_pairs_ms_median": round(float(np.median(values)), 3),
        "on_pairs_ms_p95": round(float(np.percentile(values, 95)), 3),
        "on_pairs_ms_max": round(float(values.max()), 3),
    }


def replay(session_dir: str | Path, *, root: str | Path, start_s: float = 0.0, end_s: float | None = None,
           speed: float = 1.0, config: MeasurementConfig | None = None, session_kwargs: Mapping | None = None,
           should_stop: Callable[[], bool] | None = None, on_session: Callable[[MeasurementSession], None] | None = None,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> Path | None:
    """記録を流し直し、新しい計測フォルダ（``root`` の下）を返す。記録が始まらなければ None。

    ``speed`` は再生の速さ（1 で実時間、0 で待たない）。``session_kwargs`` は ``MeasurementSession`` に渡す追加の
    キーワード（ゲージの tracker など）。``on_session`` は作った ``MeasurementSession`` を呼び出し側へ渡す
    （メインスレッドがゲージの行を出すため）。
    """
    session_dir = Path(session_dir)
    source_meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    calibration = load_session_calibration(session_dir)
    if config is None:
        # 計測の子と同じく EKF の設定は環境変数から読む（MeasurementConfig の既定は環境変数を見ない）
        config = MeasurementConfig(body_mass_kg=float(source_meta.get("body_mass_kg", 65.0)),
                                   gravity_mode=source_meta.get("gravity_mode", "axis"),
                                   ekf=EkfSettings.from_env())
    measurement = MeasurementSession(
        calibration, root=Path(root), config=config,
        metadata={"replay_of": str(session_dir), "replay_from_s": start_s, "replay_to_s": end_s,
                  "replay_speed": speed},
        **dict(session_kwargs or {}),
    )
    if on_session is not None:
        on_session(measurement)
    frames = merged_frames(read_landmarks(session_dir), start_s=start_s, end_s=end_s)
    buffer = SyncBuffer()
    stop = should_stop or (lambda: False)
    samples_ms: list[float] = []
    pair_count = [0]
    wall_start = clock()
    last_flush = wall_start
    t_first = frames[0].t_capture_ns if frames else 0

    def deliver(frame: LandmarkFrame) -> None:
        if not measurement.accept_frame(frame):
            return
        buffer.push(frame)
        measurement.on_landmarks(frame)
        pairs = buffer.drain()
        if pairs:
            started = time.perf_counter()
            measurement.on_pairs(pairs)
            samples_ms.append((time.perf_counter() - started) * 1e3 / len(pairs))
            pair_count[0] += len(pairs)

    for frame in frames:
        if stop():
            measurement.stop_reason = "stop_request"
            break
        if measurement.failed.is_set():
            break
        if speed > 0:
            target = wall_start + (frame.t_capture_ns - t_first) / 1e9 / speed
            wait = target - clock()
            if wait > 0:
                sleep(wait)
        try:
            deliver(frame)
        except Exception:  # MeasurementSession が failed と理由を立ててから投げ直す。ここで止める
            break
        now = clock()
        if now - last_flush >= _FLUSH_INTERVAL_S:
            measurement.flush()
            last_flush = now
    if measurement.stop_reason is None:
        measurement.stop_reason = "failed" if measurement.failed.is_set() else "replay_end"
    try:
        measurement.close()
    except Exception:
        pass  # 理由は meta.json（error・exit_code）に残っている
    directory = measurement.directory
    if directory is not None:
        update_meta(directory, replay_timing=_timing(samples_ms, pair_count[0]))
    return None if directory is None else Path(directory)
