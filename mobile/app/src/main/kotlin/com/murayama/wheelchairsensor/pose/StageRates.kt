package com.murayama.wheelchairsensor.pose

import java.util.EnumMap
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

/**
 * 計測中にフレームが通る段階。Pixel の点が PC に 10〜15 Hz しか届かない原因を、実機の画面で切り分けるために数える。
 *
 * 前の段より少なければ、その段で減っている。
 */
enum class Stage(val label: String) {
    /** カメラが撮ったフレーム（Camera2 の撮影完了）。30 未満なら、カメラ自体が遅い（暗い室内で露光が延びる） */
    SENSOR("カメラ"),

    /** 解析に渡ったフレーム。カメラより少なければ、解析が追いつかず CameraX が古いフレームを捨てている */
    ANALYZED("解析"),

    /** 姿勢推定の結果が返ったフレーム。解析より少なければ、推論中に来たフレームを MediaPipe が捨てている */
    INFERRED("推論"),

    /** 人が写っていたフレーム。推論より少なければ、検出が外れている（写り方・明るさ）。写っていなければ送らない */
    PERSON("人"),

    /** PC へ送ったフレーム（SensorClient が数える）。人より少なければ、送信の待ち行列が溢れて捨てている */
    SENT("送信"),
}

/** 段階ごとの累計。解析・推論の結果・送信の各スレッドから数えるので AtomicLong で持つ。 */
class StageCounter {

    private val counts = EnumMap<Stage, AtomicLong>(Stage::class.java).apply {
        Stage.values().forEach { put(it, AtomicLong(0)) }
    }

    // 推論にかかった時間（フレームを受け取ってから結果が返るまで）の合計と件数
    private val latencySumNanos = AtomicLong(0)
    private val latencyCount = AtomicLong(0)

    fun mark(stage: Stage) {
        counts.getValue(stage).incrementAndGet()
    }

    /** 1 枚の推論にかかった時間。1 枚に 70 ms かかれば、推論だけで 14 fps が上限になる */
    fun recordLatency(nanos: Long) {
        latencySumNanos.addAndGet(nanos)
        latencyCount.incrementAndGet()
    }

    /** 今の累計。[overrides] は外で数えている段（送信）の累計。 */
    fun snapshot(nowNanos: Long, overrides: Map<Stage, Long> = emptyMap()): Snapshot =
        Snapshot(
            nowNanos,
            Stage.values().associateWith { overrides[it] ?: counts.getValue(it).get() },
            latencySumNanos.get(),
            latencyCount.get(),
        )

    data class Snapshot(
        val nanos: Long,
        val counts: Map<Stage, Long>,
        val latencySumNanos: Long = 0,
        val latencyCount: Long = 0,
    )

    companion object {
        /** カメラがこれより遅ければ「カメラが遅い」と示す（30 fps の 8 割。PC 側の検査と同じ） */
        const val SLOW_CAMERA_FPS = 24.0

        /** 前の段よりこの割合を下回ったら、その段で減っているとみなす */
        private const val DROP_RATIO = 0.8

        /** 2 回の読みの間の 1 秒あたりの数。 */
        fun perSecond(previous: Snapshot, current: Snapshot): Map<Stage, Double> {
            val seconds = (current.nanos - previous.nanos) / 1e9
            return Stage.values().associateWith { stage ->
                if (seconds <= 0) 0.0
                else ((current.counts[stage] ?: 0L) - (previous.counts[stage] ?: 0L)) / seconds
            }
        }

        /** 2 回の読みの間に返った結果の、推論にかかった時間の平均 [ms]。結果が無ければ null。 */
        fun latencyMs(previous: Snapshot, current: Snapshot): Double? {
            val count = current.latencyCount - previous.latencyCount
            if (count <= 0) return null
            return (current.latencySumNanos - previous.latencySumNanos) / count / 1e6
        }

        /** 前の段より 2 割以上減った最初の段。無ければ null。 */
        fun bottleneck(rates: Map<Stage, Double>): Stage? {
            val stages = Stage.values()
            for (i in 1 until stages.size) {
                val before = rates[stages[i - 1]] ?: 0.0
                val after = rates[stages[i]] ?: 0.0
                if (before > 0.0 && after < before * DROP_RATIO) return stages[i]
            }
            return null
        }

        /**
         * 画面に出す 1 行。例: 「毎秒 カメラ 15.0 → 解析 15.0 → … （カメラが遅い: …）／推論 42 ms（CPU）」
         * [latencyMs] は推論にかかった時間、[delegate] は推論に使っているもの（CPU / GPU）。
         */
        fun describe(rates: Map<Stage, Double>, latencyMs: Double? = null, delegate: String? = null): String {
            val line = Stage.values().joinToString(" → ") {
                String.format(Locale.ROOT, "%s %.1f", it.label, rates[it] ?: 0.0)
            }
            val camera = rates[Stage.SENSOR] ?: 0.0
            val drop = bottleneck(rates)
            val hint = when {
                camera > 0.0 && camera < SLOW_CAMERA_FPS -> "（カメラが遅い: 照明を明るくするか 30fps 固定にする）"
                drop != null -> "（${drop.label}で減っている）"
                else -> ""
            }
            val latency = latencyMs?.let {
                String.format(Locale.ROOT, "／推論 %.0f ms", it) + (delegate?.let { d -> "（$d）" } ?: "")
            } ?: ""
            return "毎秒 $line$hint$latency"
        }
    }
}
