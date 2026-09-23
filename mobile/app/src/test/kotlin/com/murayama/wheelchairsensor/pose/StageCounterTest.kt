package com.murayama.wheelchairsensor.pose

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 計測中の段階ごとの 1 秒あたりの数を検証する。
 *
 * 実機の Pixel 7a は PC に 10〜15 Hz しか点が届かない（2026-09-23）。姿勢推定のモデルは最軽量（lite）なので、
 * 失われているのはモデルの外（カメラが 15 fps しか出していない、解析が追いつかずに捨てている、人を検出できない、
 * 送信で捨てている）と見ている。どの段で減っているかを実機の画面で見分けるための数え方。
 */
class StageCounterTest {

    private fun counter(vararg marks: Pair<Stage, Int>): StageCounter {
        val counter = StageCounter()
        for ((stage, times) in marks) repeat(times) { counter.mark(stage) }
        return counter
    }

    @Test
    fun `2 回の読みの差を 1 秒あたりに直す`() {
        val counter = StageCounter()
        val first = counter.snapshot(nowNanos = 1_000_000_000)
        repeat(30) { counter.mark(Stage.SENSOR) }
        repeat(15) { counter.mark(Stage.ANALYZED) }
        val rates = StageCounter.perSecond(first, counter.snapshot(nowNanos = 3_000_000_000))
        assertEquals(15.0, rates.getValue(Stage.SENSOR), 1e-9)
        assertEquals(7.5, rates.getValue(Stage.ANALYZED), 1e-9)
    }

    @Test
    fun `送信の数は外から渡した累計を使う`() {
        // 送信は SensorClient が数えている（onProgress）。二重に数えない
        val counter = StageCounter()
        val first = counter.snapshot(0, overrides = mapOf(Stage.SENT to 100L))
        val second = counter.snapshot(1_000_000_000, overrides = mapOf(Stage.SENT to 112L))
        assertEquals(12.0, StageCounter.perSecond(first, second).getValue(Stage.SENT), 1e-9)
    }

    @Test
    fun `前の段より 2 割以上減った最初の段を示す`() {
        val rates = mapOf(Stage.SENSOR to 30.0, Stage.ANALYZED to 29.0, Stage.INFERRED to 15.0,
            Stage.PERSON to 14.0, Stage.SENT to 14.0)
        assertEquals(Stage.INFERRED, StageCounter.bottleneck(rates))
        assertNull(StageCounter.bottleneck(Stage.values().associateWith { 30.0 }))
    }

    @Test
    fun `カメラ自体が遅いときはそれを先に示す`() {
        val rates = mapOf(Stage.SENSOR to 15.0, Stage.ANALYZED to 15.0, Stage.INFERRED to 15.0,
            Stage.PERSON to 14.0, Stage.SENT to 14.0)
        val text = StageCounter.describe(rates)
        assertTrue(text, text.startsWith("毎秒 カメラ 15.0 → 解析 15.0"))
        assertTrue(text, text.contains("カメラが遅い"))
    }

    @Test
    fun `時間が進んでいなければ 0 にする`() {
        val counter = counter(Stage.SENSOR to 5)
        val snapshot = counter.snapshot(1_000)
        assertEquals(0.0, StageCounter.perSecond(snapshot, snapshot).getValue(Stage.SENSOR), 0.0)
    }
}

/**
 * 推論にかかる時間（フレームを受け取ってから結果が返るまで）を検証する。
 *
 * 1 枚に 70 ms かかれば、推論だけで 14 fps が上限になる。CPU と GPU の推論を実機で比べるために出す。
 */
class InferenceTimeTest {

    @Test
    fun `2 回の読みの間に返った結果の平均を ms で出す`() {
        val counter = StageCounter()
        val first = counter.snapshot(0)
        counter.recordLatency(20_000_000)
        counter.recordLatency(40_000_000)
        val second = counter.snapshot(1_000_000_000)
        assertEquals(30.0, StageCounter.latencyMs(first, second)!!, 1e-9)
        assertNull("結果が無い間は出さない", StageCounter.latencyMs(second, counter.snapshot(2_000_000_000)))
    }

    @Test
    fun `画面の行に推論の時間を添える`() {
        val rates = Stage.values().associateWith { 30.0 }
        val text = StageCounter.describe(rates, latencyMs = 42.4, delegate = "GPU")
        assertTrue(text, text.contains("推論 42 ms（GPU）"))
    }
}
