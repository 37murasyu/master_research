package com.murayama.wheelchairsensor.capture

import com.murayama.wheelchairsensor.net.Protocol

/**
 * 受け取った撮影要求を溜め、**どのフレームで応えるか**を決める。
 *
 * 応えるのは姿勢推定に使っているのと同じフレーム。別に静止画を撮ると、解像度や
 * 画角の切り出し、焦点の扱いが計測時とずれ、校正が実際の映像と合わなくなる。
 *
 * - 目標時刻が無ければ、要求を受けた後に撮った最初のフレームで応える
 * - 目標時刻（PC 時計）があれば、端末時計へ換算し、その半フレーム前以降の最初のフレームで応える
 * - 応えられないまま [expireNanos] たった要求は捨てる。PC は 1 秒で諦めて次を出すので、
 *   後から応えても無関係な瞬間の画像になる
 *
 * 受信（OkHttp のスレッド）と取り出し（解析のスレッド）が別なので、中はロックで守る。
 * Android に依存しないので JVM のユニットテストで確かめられる。
 */
class CaptureRequests(
    private val maxPending: Int = 4,
    private val expireNanos: Long = 2_000_000_000L,
    private val halfFrameNanos: Long = 16_666_667L,
) {

    private data class Pending(val request: Protocol.CaptureRequest, val receivedDeviceNanos: Long)

    private val pending = ArrayDeque<Pending>()

    fun add(request: Protocol.CaptureRequest, receivedDeviceNanos: Long) {
        synchronized(pending) {
            pending.addLast(Pending(request, receivedDeviceNanos))
            while (pending.size > maxPending) pending.removeFirst()
        }
    }

    /**
     * 端末時計 [frameDeviceNanos] に撮ったフレームで応えるべき要求を取り出す。
     *
     * @param offsetNanos 端末時計に足すと PC 時計になる差（TimeSync.offsetNanos）。
     */
    fun takeDue(frameDeviceNanos: Long, offsetNanos: Long): List<Protocol.CaptureRequest> {
        synchronized(pending) {
            if (pending.isEmpty()) return emptyList()
            val due = mutableListOf<Protocol.CaptureRequest>()
            val iterator = pending.iterator()
            while (iterator.hasNext()) {
                val item = iterator.next()
                if (frameDeviceNanos - item.receivedDeviceNanos > expireNanos) {
                    iterator.remove()
                    continue
                }
                val readyFrom = item.request.atNanosPcClock
                    ?.let { it - offsetNanos - halfFrameNanos }
                    ?: item.receivedDeviceNanos
                if (frameDeviceNanos >= readyFrom) {
                    due.add(item.request)
                    iterator.remove()
                }
            }
            return due
        }
    }

    fun pendingCount(): Int = synchronized(pending) { pending.size }

    fun clear() = synchronized(pending) { pending.clear() }
}
