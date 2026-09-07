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

    # 位相差は平滑値で見る。瞬時値 1 サンプルはジッタの分散をそのまま拾うため、
    # 単発の値で良否を決めてはいけない（±15ms を両系統に入れると差分の標準偏差は
    # 約 21ms になり、1 フレーム 33ms を超える回が普通に出る）。
    assert stats["mean_role_skew_ms"] < 33.0, (
        f"平均位相差が 1 フレームを超えている: {stats['mean_role_skew_ms']:.1f}ms "
        f"(最大 {stats['max_role_skew_ms']:.1f}ms)"
    )


def test_full_chain_from_phones_to_torques():
    """模擬端末 -> サーバ -> 同期バッファ -> 三角測量 -> 逆動力学 の通し。

    値の物理的な妥当性は見ない。mock_sender の合成動作は任意の往復運動で
    視差もごく小さく、3D 再構成が退化するため、トルクの数値に意味は無い。
    ここで確かめるのは配管が通っていること。実際の値は実機で確認する。
    """
    import numpy as np

    from app.runners.network_measure import MeasurementConfig, NetworkMeasurement

    width, height = 1280, 720
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
    intrinsics = np.array([[900.0, 0.0, width / 2], [0.0, 900.0, height / 2], [0.0, 0.0, 1.0]])
    p_left = intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p_right = intrinsics @ np.hstack([np.eye(3), np.array([[-0.5], [0.0], [0.0]])])

    measurement = NetworkMeasurement(
        p_left, p_right, pose_keypoints, MeasurementConfig(body_mass_kg=60.0)
    )
    processed: list = []

    async def scenario():
        server = LandmarkServer(
            host="127.0.0.1",
            port=0,
            buffer=SyncBuffer(target_hz=30.0, window_sec=2.0, max_gap_ms=100.0),
            on_pairs=lambda pairs: processed.extend(
                r for r in (measurement.process(p) for p in pairs) if r is not None
            ),
        )
        await server.start()
        try:
            url = f"ws://127.0.0.1:{server.port}"
            phones = [
                MockPhone(url, "cam0", fps=30.0, jitter_ms=5.0, loss=0.0, seed=1),
                MockPhone(url, "cam1", fps=30.0, jitter_ms=5.0, loss=0.0, seed=2),
            ]
            await asyncio.gather(*(phone.run(3.0) for phone in phones))
            await asyncio.sleep(0.3)
            for pair in server.buffer.drain():
                result = measurement.process(pair)
                if result is not None:
                    processed.append(result)
        finally:
            await server.stop()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))

    assert processed, "1 フレームも処理されていない"

    with_torque = [r for r in processed if r.local_torques]
    assert with_torque, "トルクが 1 フレームも計算されていない"

    for result in with_torque:
        for name, vector in result.local_torques.items():
            assert np.all(np.isfinite(vector)), f"{name} に非有限値が混入"

    assert measurement.cycle_count > 0, "サイクルが 1 回も検出されていない"
