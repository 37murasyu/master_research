package com.murayama.wheelchairsensor.net

import android.os.SystemClock

/**
 * 端末の時計を PC の時計に合わせる。NTP と同じ往復測定を使う。
 *
 * なぜ必要か。三角測量を壊すのは 2 台のカメラの**相対的なずれ**であって、
 * 両方が揃って遅れること自体は害がない。したがって「時刻を刻んで送り、
 * PC 側が時刻でペアリングする」ようにすれば、1 秒程度のラグは許容できる。
 *
 * PC を時刻サーバにするので、2 台の端末が同一の PC 時計に揃い、
 * 結果として端末どうしも揃う。外部 NTP に頼らないのが要点。
 *
 * 時計には [SystemClock.elapsedRealtimeNanos] を使う。壁時計
 * （[System.currentTimeMillis]）は自動時刻合わせで飛ぶことがあり、
 * 計測中に時間が巻き戻ると PC 側のペアリングが壊れるため。
 */
class TimeSync {

    /** 端末時計に足すと PC 時計になる差分。 */
    @Volatile
    var offsetNanos: Long = 0
        private set

    /** 採用したサンプルの往復遅延。同期品質の目安として画面に出す。 */
    @Volatile
    var roundTripNanos: Long = 0
        private set

    @Volatile
    var isSynchronized: Boolean = false
        private set

    /** 端末の単調時計。 */
    fun deviceNanos(): Long = SystemClock.elapsedRealtimeNanos()

    /** 撮影時刻を PC 時計に直す。送信する `t_capture_ns` はこの値。 */
    fun toPcClock(deviceNanos: Long): Long = deviceNanos + offsetNanos

    /**
     * 1 回の往復測定から時計ずれと往復遅延を求める。
     *
     * t1: 端末が送信した時刻（端末時計）
     * t2: PC が受信した時刻（PC 時計）
     * t3: PC が返信した時刻（PC 時計）
     * t4: 端末が受信した時刻（端末時計）
     *
     * 往路と復路の遅延が等しいと仮定して、ずれを片道分ずつ打ち消す。
     * RTT からサーバ内の処理時間 (t3-t2) を除くのは、その間は
     * ネットワークを飛んでいないため。
     */
    fun measure(t1: Long, t2: Long, t3: Long, t4: Long): Sample {
        val offset = ((t2 - t1) + (t3 - t4)) / 2
        val rtt = (t4 - t1) - (t3 - t2)
        return Sample(offsetNanos = offset, roundTripNanos = rtt)
    }

    /**
     * 複数回の測定から最も信頼できるものを採用する。
     *
     * RTT が最小のサンプルを採る。平均を取らないのは、一時的な輻輳による
     * 外れ値に引きずられるため。往復が速かった回ほど「往路と復路が等しい」
     * という仮定からのずれが小さい。
     */
    fun adopt(samples: List<Sample>): Sample? {
        val best = samples.minByOrNull { it.roundTripNanos } ?: return null
        offsetNanos = best.offsetNanos
        roundTripNanos = best.roundTripNanos
        isSynchronized = true
        return best
    }

    fun reset() {
        offsetNanos = 0
        roundTripNanos = 0
        isSynchronized = false
    }

    data class Sample(val offsetNanos: Long, val roundTripNanos: Long) {
        val roundTripMillis: Double get() = roundTripNanos / 1_000_000.0
        val offsetMillis: Double get() = offsetNanos / 1_000_000.0
    }

    companion object {
        /** 起動時に行う往復測定の回数。 */
        const val INITIAL_SAMPLES = 20

        /** 再同期の間隔。端末の時計は少しずつドリフトする。 */
        const val RESYNC_INTERVAL_MS = 30_000L
    }
}
