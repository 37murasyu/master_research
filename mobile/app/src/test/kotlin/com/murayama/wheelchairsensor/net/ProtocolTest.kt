package com.murayama.wheelchairsensor.net

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * PC 側 `app/net/protocol.py` との電文の一致を守るテスト。
 *
 * ここがずれると、受信側は ProtocolError で黙って捨てるだけになる。
 * 実機を繋いでから気づくと切り分けが難しいので、フィールド名・型・件数を固定する。
 */
class ProtocolTest {

    private fun dummyLandmarks(count: Int = Protocol.LANDMARK_COUNT): List<FloatArray> =
        List(count) { floatArrayOf(0.5f, 0.25f, 0.0f, 0.9f) }

    @Test
    fun `landmarks の電文が PC 側の想定する形になっている`() {
        val json = JSONObject(
            Protocol.landmarks(
                role = Protocol.ROLE_CAM0,
                seq = 42L,
                captureNanosPcClock = 1_725_699_123_456_789_000L,
                width = 1280,
                height = 720,
                landmarks = dummyLandmarks(),
            )
        )

        assertEquals("landmarks", json.getString("type"))
        assertEquals("cam0", json.getString("role"))
        assertEquals(42L, json.getLong("seq"))
        assertEquals(1_725_699_123_456_789_000L, json.getLong("t_capture_ns"))
        assertEquals(1280, json.getInt("w"))
        assertEquals(720, json.getInt("h"))
        assertEquals(Protocol.LANDMARK_COUNT, json.getJSONArray("lm").length())
    }

    @Test
    fun `t_capture_ns がナノ秒の精度を保つ`() {
        // 浮動小数で扱うと 2^53 を超えた時点で精度が落ちる。整数のまま運ぶこと。
        val nanos = 1_725_699_123_456_789_123L
        val json = JSONObject(
            Protocol.landmarks(Protocol.ROLE_CAM1, 0, nanos, 1280, 720, dummyLandmarks())
        )
        assertEquals(nanos, json.getLong("t_capture_ns"))
    }

    @Test
    fun `各ランドマークは 4 要素`() {
        val lm = JSONObject(
            Protocol.landmarks(Protocol.ROLE_CAM0, 0, 1L, 1280, 720, dummyLandmarks())
        ).getJSONArray("lm")
        for (i in 0 until lm.length()) {
            assertEquals("lm[$i] は (x, y, z, visibility)", 4, lm.getJSONArray(i).length())
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ランドマーク数が違えば送信前に弾く`() {
        // 33 点でないと PC 側が 1 フレームまるごと捨てる。手前で気づけるようにする。
        Protocol.landmarks(Protocol.ROLE_CAM0, 0, 1L, 1280, 720, dummyLandmarks(count = 5))
    }

    @Test
    fun `hello にプロトコル版と役割が入る`() {
        val json = JSONObject(Protocol.hello("cam1", "Pixel 8", "abc123"))
        assertEquals("hello", json.getString("type"))
        assertEquals(Protocol.VERSION, json.getInt("v"))
        assertEquals("cam1", json.getString("role"))
        assertEquals("Pixel 8", json.getString("device"))
        assertEquals("abc123", json.getString("session"))
    }

    @Test
    fun `sync_req は端末時計をそのまま載せる`() {
        val json = JSONObject(Protocol.syncRequest(1234567890123L))
        assertEquals("sync_req", json.getString("type"))
        assertEquals(1234567890123L, json.getLong("t1"))
    }

    @Test
    fun `sync_res を読み取れる`() {
        val raw = """{"type":"sync_res","t1":1000,"t2":1050,"t3":1060}"""
        val parsed = Protocol.parseSyncResponse(raw)
        assertTrue(parsed != null)
        assertEquals(1000L, parsed!!.t1)
        assertEquals(1050L, parsed.t2)
        assertEquals(1060L, parsed.t3)
    }

    @Test
    fun `想定外の電文は null を返して落ちない`() {
        // 無線では壊れた受信が起こりうる。1 通で計測を止めない。
        assertEquals(null, Protocol.parseSyncResponse("これはJSONではない"))
        assertEquals(null, Protocol.parseSyncResponse("""{"type":"landmarks"}"""))
        assertEquals(null, Protocol.parseSyncResponse("{}"))
    }

    @Test
    fun `role の妥当性を判定できる`() {
        assertTrue(Protocol.isValidRole("cam0"))
        assertTrue(Protocol.isValidRole("cam1"))
        assertTrue(!Protocol.isValidRole("cam9"))
        assertTrue(!Protocol.isValidRole(null))
    }
}
