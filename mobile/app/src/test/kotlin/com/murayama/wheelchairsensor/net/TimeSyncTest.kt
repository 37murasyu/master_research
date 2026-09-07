package com.murayama.wheelchairsensor.net

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * 時刻同期の計算を検証する。PC 側 `compute_clock_offset` と同じ式であること。
 */
class TimeSyncTest {

    private val sync = TimeSync()

    @Test
    fun `往復が対称なら時計ずれは 0`() {
        val s = sync.measure(t1 = 1000, t2 = 1050, t3 = 1050, t4 = 1100)
        assertEquals(0L, s.offsetNanos)
        assertEquals(100L, s.roundTripNanos)
    }

    @Test
    fun `端末が遅れている分を検出できる`() {
        // PC 時刻 = 端末時刻 + 500、往復 100
        val s = sync.measure(t1 = 1000, t2 = 1550, t3 = 1550, t4 = 1100)
        assertEquals(500L, s.offsetNanos)
    }

    @Test
    fun `サーバ内の処理時間は往復遅延から除く`() {
        val s = sync.measure(t1 = 1000, t2 = 1040, t3 = 1060, t4 = 1100)
        assertEquals(80L, s.roundTripNanos)
    }

    @Test
    fun `RTT が最小のサンプルを採用する`() {
        // 平均だと一時的な輻輳の外れ値に引きずられる。最小 RTT を採る。
        val samples = listOf(
            TimeSync.Sample(offsetNanos = 900, roundTripNanos = 1000),
            TimeSync.Sample(offsetNanos = 500, roundTripNanos = 100),
            TimeSync.Sample(offsetNanos = 700, roundTripNanos = 500),
        )
        val adopted = sync.adopt(samples)
        assertEquals(500L, adopted?.offsetNanos)
        assertEquals(500L, sync.offsetNanos)
    }

    @Test
    fun `同期前は補正が恒等`() {
        val fresh = TimeSync()
        assertEquals(12345L, fresh.toPcClock(12345L))
    }

    @Test
    fun `同期後は撮影時刻が PC 時計に乗る`() {
        sync.adopt(listOf(TimeSync.Sample(offsetNanos = 1_000_000, roundTripNanos = 10)))
        assertEquals(12345L + 1_000_000L, sync.toPcClock(12345L))
    }
}
