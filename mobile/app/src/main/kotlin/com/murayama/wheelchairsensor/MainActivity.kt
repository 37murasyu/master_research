package com.murayama.wheelchairsensor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.provider.Settings
import android.os.Looper
import android.os.SystemClock
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.murayama.wheelchairsensor.camera.CameraPurpose
import com.murayama.wheelchairsensor.camera.CameraSetup
import com.murayama.wheelchairsensor.databinding.ActivityMainBinding
import com.murayama.wheelchairsensor.net.ConnectionTarget
import com.murayama.wheelchairsensor.net.SensorClient
import com.murayama.wheelchairsensor.net.TimeSync
import com.murayama.wheelchairsensor.net.Protocol
import com.murayama.wheelchairsensor.net.DeviceIdentity
import com.murayama.wheelchairsensor.capture.CaptureRequests
import com.murayama.wheelchairsensor.capture.JpegResponder
import java.util.concurrent.atomic.AtomicLong
import com.murayama.wheelchairsensor.pose.PoseAnalyzer
import com.murayama.wheelchairsensor.pose.Stage
import com.murayama.wheelchairsensor.pose.StageCounter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * 端末を「無線の姿勢センサ」にする画面。
 *
 * 流れ:
 *   1. PC が表示する QR を読み取り、接続先と役割（cam0 / cam1）を得る
 *   2. WebSocket で繋ぎ、往復測定で PC の時計に合わせる
 *   3. カメラを開いて MediaPipe を回し、ランドマークを時刻付きで送り続ける
 *
 * ランドマークを連続送信し、PC の要求時だけ解析フレームの JPEG も返す。
 */
class MainActivity : AppCompatActivity(), SensorClient.Listener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var cameraSetup: CameraSetup

    private val captureRequests = CaptureRequests()
    private val jpegExecutor = Executors.newSingleThreadExecutor()
    private val imagesSent = AtomicLong(0)
    private val framesSent = AtomicLong(0)
    private val framesDropped = AtomicLong(0)
    private val jpegResponder by lazy {
        JpegResponder(jpegExecutor, send = { req, nanos, w, h, jpeg ->
            if (client.sendCalibrationFrame(req.id, nanos, w, h, jpeg)) imagesSent.incrementAndGet()
        })
    }
    private val timeSync = TimeSync()
    private val client by lazy { SensorClient(timeSync, this) }
    private val mainHandler = Handler(Looper.getMainLooper())

    private var poseAnalyzer: PoseAnalyzer? = null
    // 段階ごとの数（カメラ・解析・推論・人・送信）。計測中の画面に 1 秒あたりの数を出し、遅い段を見分ける
    @Volatile
    private var stageCounter = StageCounter()
    private var lastStages: StageCounter.Snapshot? = null
    // 検出器はネイティブモデルを抱えるので、押すたびに作らず 1 個を持ち回す。
    private var barcodeScanner: BarcodeScanner? = null
    private var mode = Mode.IDLE

    // -- 自動の再接続 ------------------------------------------------------
    // QR は最初の 1 回だけ読む。以後は覚えた接続先へ、切れたら数秒ごとにつなぎ直す。
    // PC 側は session をその PC に保存して使い回す（app/hybrid/session.py）ので、
    // PC 側のツールを起動し直しても同じ接続先のまま受け入れられる。
    private val prefs by lazy { getSharedPreferences(PREFS, MODE_PRIVATE) }
    private var lastTarget: ConnectionTarget? = null
    /** 切れたらつなぎ直すか。「切断」を押す・QR を読み直す・PC に断られると止める。 */
    private var autoReconnect = false
    /** つなぎ直しの途中か。エラーのたびにトーストを出さず、画面の案内に留める。 */
    private var reconnecting = false

    private enum class Mode { IDLE, SCANNING, STREAMING }

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startScanning()
        } else {
            binding.detailText.text = getString(R.string.camera_permission_required)
        }
    }

    // -- ライフサイクル -----------------------------------------------------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 計測中に画面が消えると CameraX がフレームを止め、その端末の
        // ランドマークが途絶える。PC 側は片方を待ち続けてペアを出さなくなり、
        // ログにも何も出ない。マニフェストの android:keepScreenOn は
        // View の属性でありアクティビティには効かないので、ここで立てる。
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        analysisExecutor = Executors.newSingleThreadExecutor()
        cameraSetup = CameraSetup(this, binding.preview, analysisExecutor)

        binding.scanButton.setOnClickListener { ensureCameraThenScan() }
        binding.disconnectButton.setOnClickListener { stopStreaming() }
        // 30fps 固定（既定はオン）。計測中に切り替えると、その場でカメラを開き直して比べられる
        binding.pinFpsCheck.isChecked = prefs.getBoolean(KEY_PIN_FPS, true)
        binding.pinFpsCheck.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(KEY_PIN_FPS, checked).apply()
            val analyzer = poseAnalyzer
            if (mode == Mode.STREAMING && analyzer != null) {
                lastStages = null
                bindCamera(CameraPurpose.MEASURE, analyzer)
            }
        }
        // GPU 推論（既定はオフ）。計測中に切り替えると、推定器を作り直してその場で比べられる
        binding.gpuCheck.isChecked = prefs.getBoolean(KEY_GPU, false)
        binding.gpuCheck.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean(KEY_GPU, checked).apply()
            if (mode == Mode.STREAMING) {
                val old = poseAnalyzer
                val analyzer = createAnalyzer()
                poseAnalyzer = analyzer
                bindCamera(CameraPurpose.MEASURE, analyzer)
                old?.close()
            }
        }

        // QR 読み取り中は、タップした場所にピントを合わせる（計測中は CameraSetup が無視する）
        binding.preview.setOnTouchListener { view, event ->
            if (event.action == MotionEvent.ACTION_UP && mode == Mode.SCANNING) {
                cameraSetup.focusAt(event.x, event.y)
                view.performClick()
            }
            true
        }

        // 前に QR を読んだ PC があれば、そこへ自動でつなぐ（計測にはカメラの許可が要る）
        val saved = prefs.getString(KEY_LAST_URL, null)?.let { ConnectionTarget.parse(it).getOrNull() }
        val cameraGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        if (saved != null && cameraGranted) {
            connect(saved)
            binding.detailText.text = "前回の PC（${saved.host}）へ接続します。別の PC なら「PCのQRコードを読み取る」"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        client.disconnect()
        poseAnalyzer?.close()
        barcodeScanner?.close()
        cameraSetup.stop()
        analysisExecutor.shutdown()
        captureRequests.clear()
        jpegExecutor.shutdown()
    }

    // -- QR 読み取り --------------------------------------------------------
    private fun ensureCameraThenScan() {
        val granted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (granted) startScanning() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startScanning() {
        autoReconnect = false
        reconnecting = false
        mainHandler.removeCallbacks(reconnect)
        mode = Mode.SCANNING
        binding.statusText.text = "QR を読み取ってください"
        binding.detailText.text = SCAN_HINT

        val scanner = barcodeScanner ?: BarcodeScanning.getClient().also { barcodeScanner = it }
        bindCamera(
            CameraPurpose.SCAN_QR,
            ImageAnalysis.Analyzer { image ->
                processBarcode(image, scanner)
            }
        )
        scheduleRefocus()
    }

    /**
     * 読み取れるまで、数秒ごとに画面の中央へピントを合わせ直す。
     *
     * 端末を QR に近づけたり離したりすると、連続 AF が追いつかずにぼけたまま止まることがある。
     */
    private fun scheduleRefocus() {
        mainHandler.removeCallbacks(refocus)
        mainHandler.postDelayed(refocus, REFOCUS_INTERVAL_MS)
    }

    private val refocus = Runnable {
        if (mode == Mode.SCANNING) {
            cameraSetup.focusCenter()
            scheduleRefocus()
        }
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun processBarcode(proxy: ImageProxy, scanner: BarcodeScanner) {
        val mediaImage = proxy.image
        if (mediaImage == null || mode != Mode.SCANNING) {
            proxy.close()
            return
        }

        val input = InputImage.fromMediaImage(mediaImage, proxy.imageInfo.rotationDegrees)
        scanner.process(input)
            .addOnSuccessListener { barcodes -> onBarcodes(barcodes) }
            .addOnCompleteListener { proxy.close() }
    }

    private fun onBarcodes(barcodes: List<Barcode>) {
        if (mode != Mode.SCANNING) return
        val raw = barcodes.firstNotNullOfOrNull { it.rawValue } ?: return

        ConnectionTarget.parse(raw)
            .onSuccess { target ->
                mode = Mode.IDLE  // 二重に読み取らないよう即座に抜ける
                connect(target)
            }
            .onFailure { error ->
                // 何が悪いのかを画面に出す。現場で「繋がらない」だけだと詰む。
                binding.detailText.text = error.message
            }
    }

    // -- 接続と送信 ---------------------------------------------------------
    private fun connect(target: ConnectionTarget) {
        lastTarget = target
        autoReconnect = true
        binding.statusText.text = getString(R.string.status_connecting)
        binding.detailText.text = "${target.role} として ${target.host}:${target.port} へ接続します"
        client.connect(target, deviceName(), deviceId())
    }

    private val reconnect = Runnable {
        val target = lastTarget
        if (autoReconnect && target != null) client.connect(target, deviceName(), deviceId())
    }

    override fun onConnectionLost(retryable: Boolean) = runOnUiThread {
        if (autoReconnect && retryable && lastTarget != null) {
            reconnecting = true
            mainHandler.removeCallbacks(reconnect)
            mainHandler.postDelayed(reconnect, RECONNECT_INTERVAL_MS)
        } else {
            // PC がはっきり断った。理由を画面に残し、QR を読み直してもらう
            autoReconnect = false
            reconnecting = false
        }
    }

    private fun startStreaming() {
        mode = Mode.STREAMING
        poseAnalyzer?.close()

        captureRequests.clear()
        imagesSent.set(0)
        framesSent.set(0)
        framesDropped.set(0)
        val analyzer = createAnalyzer()
        poseAnalyzer = analyzer
        bindCamera(CameraPurpose.MEASURE, analyzer)
        scheduleResync()
        mainHandler.removeCallbacks(statusTicker)
        mainHandler.post(statusTicker)
    }

    /** 計測用の推定器を作る。段階の数え直しも始める（GPU の切り替えでも使う） */
    private fun createAnalyzer(): PoseAnalyzer {
        val counter = StageCounter()
        stageCounter = counter
        lastStages = null
        return PoseAnalyzer(
            this,
            captureSink = { bitmap, nanos ->
                val due = captureRequests.takeDue(nanos, timeSync.offsetNanos)
                if (due.isNotEmpty()) jpegResponder.offer(bitmap, nanos, due)
            },
            counter = counter,
            useGpu = prefs.getBoolean(KEY_GPU, false),
        ) { detection ->
            client.sendLandmarks(
                captureDeviceNanos = detection.captureDeviceNanos,
                width = detection.width,
                height = detection.height,
                landmarks = detection.landmarks,
            )
        }
    }

    private fun stopStreaming() {
        autoReconnect = false
        reconnecting = false
        mode = Mode.IDLE
        mainHandler.removeCallbacksAndMessages(null)
        client.disconnect()
        // カメラを先に止める。推定器を先に閉じると、止まる前のフレームが閉じかけの推定器へ渡る
        cameraSetup.stop()
        poseAnalyzer?.close()
        poseAnalyzer = null
        binding.disconnectButton.isEnabled = false
        binding.scanButton.isEnabled = true
    }

    /** 端末の時計は少しずつずれる。定期的に測り直す。 */
    private fun scheduleResync() {
        mainHandler.postDelayed({
            if (mode == Mode.STREAMING) {
                client.resync()
                scheduleResync()
            }
        }, TimeSync.RESYNC_INTERVAL_MS)
    }

    private fun bindCamera(purpose: CameraPurpose, analyzer: ImageAnalysis.Analyzer) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val measuring = purpose == CameraPurpose.MEASURE
                cameraSetup.start(
                    future.get(), purpose, analyzer,
                    pinFrameRate = measuring && prefs.getBoolean(KEY_PIN_FPS, true),
                    onSensorFrame = if (measuring) ({ stageCounter.mark(Stage.SENSOR) }) else null,
                ) {
                    if (purpose == CameraPurpose.SCAN_QR) {
                        runOnUiThread { binding.detailText.text = SCAN_HINT }
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    binding.statusText.text = getString(R.string.status_error)
                    binding.detailText.text = "カメラを開けませんでした: ${e.message}"
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * 端末 ID（ANDROID_ID のハッシュ）。取れなければ null で、名乗りに ID を付けない。
     *
     * 例外にすると接続の瞬間にアプリが落ちる。ID が無くてもライブ表示は使え、
     * 校正と計測は PC 側が「端末 ID を取得できません」と案内して止める。
     */
    private fun deviceId(): String? =
        Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
            ?.let(DeviceIdentity::hash)

    override fun onCaptureRequested(request: Protocol.CaptureRequest) {
        captureRequests.add(request, timeSync.deviceNanos())
    }

    private fun deviceName(): String = "${Build.MANUFACTURER} ${Build.MODEL}"

    // -- SensorClient.Listener ---------------------------------------------
    override fun onState(state: SensorClient.State, detail: String?) = runOnUiThread {
        binding.statusText.text = when (state) {
            SensorClient.State.IDLE -> getString(R.string.status_idle)
            SensorClient.State.CONNECTING -> getString(R.string.status_connecting)
            SensorClient.State.SYNCING -> getString(R.string.status_syncing)
            SensorClient.State.STREAMING -> getString(R.string.status_streaming)
            SensorClient.State.ERROR -> getString(R.string.status_error)
        }
        detail?.let { binding.detailText.text = it }

        binding.disconnectButton.isEnabled = state != SensorClient.State.IDLE
        binding.scanButton.isEnabled = state == SensorClient.State.IDLE ||
            state == SensorClient.State.ERROR

        if (state == SensorClient.State.STREAMING) {
            reconnecting = false
            // 送信まで進んだ接続先だけを覚える。PC に断られた QR を覚えると、次に開いたときも断られる
            lastTarget?.let { prefs.edit().putString(KEY_LAST_URL, it.url).apply() }
        }
        if (state == SensorClient.State.STREAMING && mode != Mode.STREAMING) {
            startStreaming()
        }
        if (reconnecting && (state == SensorClient.State.ERROR || state == SensorClient.State.IDLE)) {
            binding.statusText.text = "PC を待っています"
            binding.detailText.text = "${lastTarget?.host ?: "PC"} へ ${RECONNECT_INTERVAL_MS / 1000} 秒ごとにつなぎ直します。" +
                "PC 側でライブ表示・校正・計測のどれかを起動してください"
        } else if (state == SensorClient.State.ERROR) {
            Toast.makeText(this, detail ?: "接続に失敗しました", Toast.LENGTH_LONG).show()
        }
    }

    override fun onSynchronized(sample: TimeSync.Sample) = runOnUiThread {
        binding.detailText.text = String.format(
            "時刻同期 完了  往復 %.2f ms / 時計ずれ %.2f ms",
            sample.roundTripMillis, sample.offsetMillis,
        )
    }

    override fun onProgress(sent: Long, dropped: Long) {
        // 表示は 1 秒ごとの updateStreamingStatus に任せる（毎フレーム更新すると UI スレッドを圧迫する）
        framesSent.set(sent)
        framesDropped.set(dropped)
    }

    /**
     * 計測中の表示。人が写っているかを必ず出す。
     *
     * 点は人が写っている間しか送らないので、「送信中」とだけ出すと、人を写す必要があることが
     * 伝わらない（端末を手に持って顔に近づけたまま、点が 1 件も届かないことがあった）。
     */
    private fun updateStreamingStatus() {
        val analyzer = poseAnalyzer ?: return
        if (client.state != SensorClient.State.STREAMING) return  // 同期中・エラーの表示は onState に任せる
        val sincePerson = (SystemClock.elapsedRealtimeNanos() - analyzer.lastPersonNanos) / 1e9
        val size = if (analyzer.lastFrameWidth > 0) "${analyzer.lastFrameWidth}x${analyzer.lastFrameHeight}" else "—"
        // 段階ごとの 1 秒あたりの数（前回の表示からの差）。送信は SensorClient の累計を使う
        val snapshot = stageCounter.snapshot(
            SystemClock.elapsedRealtimeNanos(), overrides = mapOf(Stage.SENT to framesSent.get())
        )
        val rates = lastStages?.let {
            StageCounter.describe(
                StageCounter.perSecond(it, snapshot),
                latencyMs = StageCounter.latencyMs(it, snapshot),
                delegate = analyzer.delegateName,
            )
        }
        lastStages = snapshot
        val rateLine = rates?.let { "\n$it" } ?: ""
        if (sincePerson < PERSON_TIMEOUT_SEC) {
            binding.statusText.text = "送信中（人を検出中）"
            binding.detailText.text = "送信 ${framesSent.get()} フレーム / 画像 ${imagesSent.get()} 枚 / " +
                "破棄 ${framesDropped.get()} / " +
                String.format("往復 %.2f ms", timeSync.roundTripNanos / 1_000_000.0) + " / 解像度 $size" + rateLine
        } else {
            binding.statusText.text = "送信中（人が写っていません）"
            binding.detailText.text = "Pixel を机などに置き、1.5〜2 m 離れて、肩から手までが写るように向けてください。" +
                "点は人が写っている間だけ送ります" + rateLine
        }
    }

    private val statusTicker = object : Runnable {
        override fun run() {
            if (mode != Mode.STREAMING) return
            updateStreamingStatus()
            mainHandler.postDelayed(this, STATUS_INTERVAL_MS)
        }
    }

    companion object {
        /** QR 読み取り中に中央へピントを合わせ直す間隔。合わせてから連続 AF に戻るまでは 3 秒。 */
        private const val REFOCUS_INTERVAL_MS = 2_500L
        private const val PREFS = "connection"
        private const val KEY_LAST_URL = "last_url"
        /** 計測でカメラを 30fps に固定するか（CameraSetup.start の pinFrameRate） */
        private const val KEY_PIN_FPS = "pin_fps"
        /** 姿勢推定を GPU で行うか（PoseAnalyzer の useGpu） */
        private const val KEY_GPU = "gpu_inference"
        /** 切れたときにつなぎ直す間隔。 */
        private const val RECONNECT_INTERVAL_MS = 3_000L
        /** 計測中の表示を更新する間隔。 */
        private const val STATUS_INTERVAL_MS = 1_000L
        /** これより長く人を検出しなければ「人が写っていません」と出す。 */
        private const val PERSON_TIMEOUT_SEC = 1.5
        private const val SCAN_HINT =
            "PC の画面の QR に 20〜40 cm 離して向けてください。ぼけたら画面の QR をタップするとピントが合います"
    }
}
