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
        /** PC から撮影要求が届いた（OkHttp のスレッドで呼ばれる）。 */
        fun onCaptureRequested(request: Protocol.CaptureRequest) {}
    }

    enum class State { IDLE, CONNECTING, SYNCING, STREAMING, ERROR }

    private val http = OkHttpClient.Builder()
        // 無通信でも切られないようにする。姿勢が検出できない間は
        // ランドマークを送らないので、その間の無通信が長くなり得る。
        .pingInterval(15, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .build()

    @Volatile
    private var socket: WebSocket? = null
    private var target: ConnectionTarget? = null

    /**
     * 接続ごとの番号。通知が今の接続のものかをこれで見分ける。
     *
     * ソケットの同一性で比べると、`newWebSocket` の戻り値を代入し終わる前に届いた
     * onOpen を「古い接続」と取り違えて hello を送り損ねる。
     */
    @Volatile
    private var generation = 0

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
    /**
     * @param deviceId 端末ごとに変わらない識別子。PC は校正時の端末との照合に使う。
     */
    fun connect(target: ConnectionTarget, deviceName: String, deviceId: String? = null) {
        disconnect()
        this.target = target
        sequence.set(0)
        sentCount.set(0)
        droppedCount.set(0)
        timeSync.reset()

        updateState(State.CONNECTING, target.url)

        val myGeneration = generation
        val request = Request.Builder().url(target.url).build()
        socket = http.newWebSocket(request, object : WebSocketListener() {

            // 以下の通知は、今の接続のものかを必ず確かめる。QR を読み直すと古い接続の
            // 閉じた通知が新しい接続の後から届き、確かめないと新しい接続まで消してしまう。
            private fun isCurrent() = generation == myGeneration

            override fun onOpen(webSocket: WebSocket, response: Response) {
                if (!isCurrent()) return
                webSocket.send(Protocol.hello(target.role, deviceName, target.session, deviceId))
                beginSync(webSocket)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                if (!isCurrent()) return
                when (Protocol.messageType(text)) {
                    "sync_res" -> handleSyncResponse(webSocket, text)
                    "capture_req" -> Protocol.parseCaptureRequest(text)?.let(listener::onCaptureRequested)
                    else -> Unit  // 知らない電文は捨てる。PC 側が新しくても落ちない
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (!isCurrent()) return
                Log.w(TAG, "接続に失敗しました", t)
                socket = null
                updateState(State.ERROR, t.message ?: "接続に失敗しました")
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                // PC が閉じ始めた。応じて閉じ返す（そうしないと onClosed が来ない）
                webSocket.close(code, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (!isCurrent()) return
                socket = null
                if (code == NORMAL_CLOSURE || code == GOING_AWAY) {
                    updateState(State.IDLE, reason.ifBlank { null })
                } else {
                    // 役割の違い・古い QR・別の端末への交代など。理由を画面に出さないと、
                    // 使う人は「繋がらない」としか分からない。
                    updateState(State.ERROR, reason.ifBlank { "PC から切断されました（コード $code）" })
                }
            }
        })
    }

    fun disconnect() {
        // 先に番号を進め、閉じた通知が後から届いても今の状態を書き換えないようにする
        generation += 1
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

    /**
     * 撮影要求への応答を送る。時刻同期が済むまでは送らない（PC 時計に直せないため）。
     *
     * @param captureDeviceNanos 撮影時刻（端末の単調時計）。PC 時計への変換はここで行う。
     * @return 送信キューに積めたか。
     */
    fun sendCalibrationFrame(
        id: Long,
        captureDeviceNanos: Long,
        width: Int,
        height: Int,
        jpeg: ByteArray,
    ): Boolean {
        val webSocket = socket
        val role = target?.role
        if (webSocket == null || role == null || !timeSync.isSynchronized) return false
        return webSocket.send(
            Protocol.calibrationFrame(
                role = role,
                id = id,
                captureNanosPcClock = timeSync.toPcClock(captureDeviceNanos),
                width = width,
                height = height,
                jpeg = jpeg,
            )
        )
    }

    private fun updateState(next: State, detail: String? = null) {
        state = next
        listener.onState(next, detail)
    }

    companion object {
        private const val TAG = "SensorClient"

        private const val NORMAL_CLOSURE = 1000
        private const val GOING_AWAY = 1001
    }
}
