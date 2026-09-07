"""模擬端末 2 台 → サーバ → 同期バッファ の通し検証。

計画の Phase 2 合格条件をそのままテストにしたもの。

- 2 系統を別々のジッタで流し、再標本化後のペアが十分に揃うこと
- 時刻同期の精度が実用範囲に収まること

ここが通れば、Android アプリが無くても PC 側は完成させられる。
"""

from __future__ import annotations

import asyncio

from app.net.mock_sender import MockPhone
from app.net.server import LandmarkServer
from app.net.sync_buffer import SyncBuffer


async def _run_session(duration: float, jitter_ms: float, loss: float):
    pairs: list = []
    server = LandmarkServer(
        host="127.0.0.1",
        port=0,
        buffer=SyncBuffer(target_hz=30.0, window_sec=2.0, max_gap_ms=100.0),
        on_pairs=pairs.extend,
    )
    await server.start()
    try:
        url = f"ws://127.0.0.1:{server.port}"
        phones = [
            MockPhone(url, "cam0", fps=30.0, jitter_ms=jitter_ms, loss=loss, seed=1),
            MockPhone(url, "cam1", fps=30.0, jitter_ms=jitter_ms, loss=loss, seed=2),
        ]
        await asyncio.gather(*(phone.run(duration) for phone in phones))
        # 最後の数フレームがバッファに残っているので掃き出す
        await asyncio.sleep(0.2)
        pairs.extend(server.buffer.drain())
        return pairs, phones, server
    finally:
        await server.stop()


def test_two_mock_phones_produce_paired_stream():
    """理想に近い条件で、ほぼ全フレームがペアになること。"""
    pairs, phones, server = asyncio.run(
        asyncio.wait_for(_run_session(duration=2.0, jitter_ms=0.0, loss=0.0), timeout=40)
    )

    sent = min(phone.sent for phone in phones)
    assert pairs, "ペアが 1 つも出ていない"
    # グリッド再標本化なので送信数と厳密には一致しない。7 割出ていれば十分。
    assert len(pairs) >= sent * 0.7, f"ペアが少なすぎる: {len(pairs)} / 送信 {sent}"
    assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)


def test_clock_sync_is_accurate_enough_for_triangulation():
    """時刻同期の誤差が、三角測量の要求精度に対して十分小さいこと。

    手の速度を 1.5 m/s とすると、10ms のずれは 1.5cm の位置誤差。
    マーカーレス計測自体の基準誤差が 2〜3cm なので、この範囲なら影響は小さい。
    """
    _pairs, phones, _server = asyncio.run(
        asyncio.wait_for(_run_session(duration=1.0, jitter_ms=0.0, loss=0.0), timeout=40)
    )

    for phone in phones:
        assert phone.rtt_ns > 0, f"{phone.role} の同期が行われていない"
        assert phone.rtt_ns / 1e6 < 50.0, f"{phone.role} の RTT が大きすぎる"

    # 2 台が同じ PC 時計に揃っているので、互いのずれも小さいはず
    skew_ms = abs(phones[0].offset_ns - phones[1].offset_ns) / 1e6
    assert skew_ms < 10.0, f"端末間の時計ずれが {skew_ms:.1f}ms と大きい"


def test_survives_jitter_and_packet_loss():
    """劣化条件でも破綻せず、緩やかに性能が落ちること。"""
    pairs, phones, server = asyncio.run(
        asyncio.wait_for(_run_session(duration=2.0, jitter_ms=15.0, loss=0.05), timeout=40)
    )

    sent = min(phone.sent for phone in phones)
    assert pairs, "劣化条件でペアが全く出ていない"
    assert len(pairs) >= sent * 0.5, f"劣化時の歩留まりが低すぎる: {len(pairs)} / {sent}"

    stats = server.buffer.stats
    assert stats["rejected"] == 0, "未知のロールが混入している"
    # 位相差が 1 フレーム（33ms）以内に収まっていること
    assert stats["last_role_skew_ms"] < 33.0
