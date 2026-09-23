package com.murayama.wheelchairsensor.capture

import com.murayama.wheelchairsensor.net.Protocol
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 撮影要求に「どのフレームで」応えるかを検証する。
 *
 * 応えるのは姿勢推定に使っているのと同じフレーム。別に静止画を撮ると、解像度や
 * 画角の切り出し、焦点の扱いが計測時とずれ、校正が実際の映像と合わなくなる。
 */
class CaptureRequestsTest {

    private fun request(id: Long, atNanos: Long? = null) =
        Protocol.CaptureRequest(id = id, atNanosPcClock = atNanos, maxWidth = null, quality = null)

    @Test
    fun `時刻の指定が無ければ要求を受けた後の最初のフレームで応える`() {
        val requests = CaptureRequests()
        requests.add(request(1), receivedDeviceNanos = 1_000)

        assertTrue("要求より前に撮ったフレームでは応えない", requests.takeDue(900, offsetNanos = 0).isEmpty())
        assertEquals(listOf(1L), requests.takeDue(1_100, offsetNanos = 0).map { it.id })
    }

    @Test
    fun `1 つの要求には 1 回だけ応える`() {
        val requests = CaptureRequests()
        requests.add(request(1), receivedDeviceNanos = 1_000)
        requests.takeDue(1_100, offsetNanos = 0)
        assertTrue(requests.takeDue(1_200, offsetNanos = 0).isEmpty())
    }

    @Test
    fun `目標時刻は PC 時計なので端末時計へ換算して待つ`() {
        // PC 時計 = 端末時計 + 500。PC 時計の 10_000 は端末時計の 9_500
        val requests = CaptureRequests(halfFrameNanos = 0)
        requests.add(request(2, atNanos = 10_000), receivedDeviceNanos = 1_000)

        assertTrue(requests.takeDue(9_400, offsetNanos = 500).isEmpty())
        assertEquals(listOf(2L), requests.takeDue(9_500, offsetNanos = 500).map { it.id })
    }

    @Test
    fun `目標時刻の半フレーム前からは応えてよい`() {
        // 目標ちょうどのフレームは無いので、最も近いフレームで応えるための猶予
        val requests = CaptureRequests(halfFrameNanos = 16)
        requests.add(request(3, atNanos = 1_000), receivedDeviceNanos = 0)
        assertEquals(listOf(3L), requests.takeDue(985, offsetNanos = 0).map { it.id })
    }

    @Test
    fun `応えられないまま古くなった要求は捨てる`() {
        // PC 側は 1 秒で諦めて次を出す。溜めたまま後で応えると、無関係な瞬間の画像になる
        val requests = CaptureRequests(expireNanos = 2_000)
        requests.add(request(1, atNanos = 1_000_000), receivedDeviceNanos = 0)
        assertTrue(requests.takeDue(2_500, offsetNanos = 0).isEmpty())
        assertEquals(0, requests.pendingCount())
    }

    @Test
    fun `溜められる数には上限があり古いものから捨てる`() {
        val requests = CaptureRequests(maxPending = 2)
        requests.add(request(1), receivedDeviceNanos = 0)
        requests.add(request(2), receivedDeviceNanos = 0)
        requests.add(request(3), receivedDeviceNanos = 0)
        assertEquals(listOf(2L, 3L), requests.takeDue(100, offsetNanos = 0).map { it.id })
    }
}
