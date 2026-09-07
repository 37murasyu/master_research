package com.murayama.wheelchairsensor.net

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * PC の受信サーバと繋ぎ、時刻同期を済ませてからランドマークを流す。
 *
 * 接続先は PC が画面に出す QR から受け取る:
 *   `ws://<pc-ip>:<port>/?session=<id>&role=<cam0|cam1>`
 *
 * トランスポートは WebSocket（TCP）。1 秒程度のラグを許容できる前提なので、
 * TCP のヘッドオブラインブロッキングは問題にならず、パケットロスの扱いを
 * 自前で書かずに済む。
 */
class SensorClient(
    private val timeSync: TimeSync,
    private val listener: Listener,
) {

    interface Listener {
        fun onState(state: State, detail: String? = null)
        fun onSynchronized(sample: TimeSync.Sample)
        /** 送信できた累計と、送れなかった累計。画面に出す。 */
        fun onProgress(sent: Long, dropped: Long)
    }

    enum class State { IDLE, CONNECTING, SYNCING, STREAMING, ERROR }

    private val http = OkHttpClient.Builder()
        // 無通信でも切られないようにする。姿勢が検出できない間は
        // ランドマークを送らないので、その間の無通信が長くなり得る。
        .pingInterval(15, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .build()

    private var socket: WebSocket? = null
    private var target: ConnectionTarget? = null

    private val sequence = AtomicLong(0)
    private val sentCount = AtomicLong(0)
    private val droppedCount = AtomicLong(0)

    private val pendingSyncSamples = mutableListOf<TimeSync.Sample>()
    private var syncRequestsLeft = 0
    private var lastSyncSentAtDeviceNanos = 0L

    @Volatile
    var state: State = State.IDLE
        private set

    // -- 接続 ---------------------------------------------------------------
    fun connect(target: ConnectionTarget, deviceName: String) {
        disconnect()
        this.target = target
        sequence.set(0)
        sentCount.set(0)
        droppedCount.set(0)
        timeSync.reset()

        updateState(State.CONNECTING, target.url)

        val request = Request.Builder().url(target.url).build()
        socket = http.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(Protocol.hello(target.role, deviceName, target.session))
                beginSync(webSocket)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleSyncResponse(webSocket, text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "接続に失敗しました", t)
                updateState(State.ERROR, t.message ?: "接続に失敗しました")
                socket = null
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                updateState(State.IDLE, reason.ifBlank { null })
                socket = null
            }
        })
    }

    fun disconnect() {
        socket?.close(1000, "端末側から切断")
        socket = null
        timeSync.reset()
        updateState(State.IDLE)
    }

    // -- 時刻同期 -----------------------------------------------------------
    private fun beginSync(webSocket: WebSocket) {
        updateState(State.SYNCING)
        synchronized(pendingSyncSamples) {
            pendingSyncSamples.clear()
            syncRequestsLeft = TimeSync.INITIAL_SAMPLES
        }
        sendSyncRequest(webSocket)
    }

    private fun sendSyncRequest(webSocket: WebSocket) {
        lastSyncSentAtDeviceNanos = timeSync.deviceNanos()
        webSocket.send(Protocol.syncRequest(lastSyncSentAtDeviceNanos))
    }

    private fun handleSyncResponse(webSocket: WebSocket, text: String) {
        val response = Protocol.parseSyncResponse(text) ?: return
        val t4 = timeSync.deviceNanos()
        val sample = timeSync.measure(response.t1, response.t2, response.t3, t4)

        val finished: Boolean
        synchronized(pendingSyncSamples) {
            pendingSyncSamples.add(sample)
            syncRequestsLeft -= 1
            finished = syncRequestsLeft <= 0
        }

        if (!finished) {
            sendSyncRequest(webSocket)
            return
        }

        val adopted = synchronized(pendingSyncSamples) { timeSync.adopt(pendingSyncSamples.toList()) }
        if (adopted == null) {
            updateState(State.ERROR, "時刻同期に失敗しました")
            return
        }
        listener.onSynchronized(adopted)
        updateState(State.STREAMING)
    }

    /** 定期的な再同期。端末の時計は少しずつドリフトする。 */
    fun resync() {
        val webSocket = socket ?: return
        if (state != State.STREAMING) return
        beginSync(webSocket)
    }

    // -- 送信 ---------------------------------------------------------------
    /**
     * 1 フレーム分を送る。
     *
     * @param captureDeviceNanos 撮影時刻（端末の単調時計）。PC 時計への変換はここで行う。
     */
    fun sendLandmarks(
        captureDeviceNanos: Long,
        width: Int,
        height: Int,
        landmarks: List<FloatArray>,
    ) {
        val webSocket = socket
        val role = target?.role
        if (webSocket == null || role == null || !timeSync.isSynchronized) {
            droppedCount.incrementAndGet()
            return
        }

        val payload = Protocol.landmarks(
            role = role,
            seq = sequence.getAndIncrement(),
            captureNanosPcClock = timeSync.toPcClock(captureDeviceNanos),
            width = width,
            height = height,
            landmarks = landmarks,
        )

        // send は送信キューに積むだけで、溢れると false を返す。
        // 溢れたら捨てる。古いフレームを無理に送っても、PC 側は時刻で
        // ペアを組むので遅れて届いたものは使われない。
        if (webSocket.send(payload)) {
            sentCount.incrementAndGet()
        } else {
            droppedCount.incrementAndGet()
        }
        listener.onProgress(sentCount.get(), droppedCount.get())
    }

    private fun updateState(next: State, detail: String? = null) {
        state = next
        listener.onState(next, detail)
    }

    companion object {
        private const val TAG = "SensorClient"
    }
}
