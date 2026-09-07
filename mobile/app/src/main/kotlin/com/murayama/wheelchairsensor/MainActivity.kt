package com.murayama.wheelchairsensor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.murayama.wheelchairsensor.camera.CameraSetup
import com.murayama.wheelchairsensor.databinding.ActivityMainBinding
import com.murayama.wheelchairsensor.net.ConnectionTarget
import com.murayama.wheelchairsensor.net.SensorClient
import com.murayama.wheelchairsensor.net.TimeSync
import com.murayama.wheelchairsensor.pose.PoseAnalyzer
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
 * 映像は送らない。33 点で 1 フレーム約 600 バイト、30fps でも 18 KB/s。
 */
class MainActivity : AppCompatActivity(), SensorClient.Listener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var cameraSetup: CameraSetup

    private val timeSync = TimeSync()
    private val client by lazy { SensorClient(timeSync, this) }
    private val mainHandler = Handler(Looper.getMainLooper())

    private var poseAnalyzer: PoseAnalyzer? = null
    private var mode = Mode.IDLE

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

        analysisExecutor = Executors.newSingleThreadExecutor()
        cameraSetup = CameraSetup(this, binding.preview, analysisExecutor)

        binding.scanButton.setOnClickListener { ensureCameraThenScan() }
        binding.disconnectButton.setOnClickListener { stopStreaming() }
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        client.disconnect()
        poseAnalyzer?.close()
        cameraSetup.stop()
        analysisExecutor.shutdown()
    }

    // -- QR 読み取り --------------------------------------------------------
    private fun ensureCameraThenScan() {
        val granted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (granted) startScanning() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startScanning() {
        mode = Mode.SCANNING
        binding.statusText.text = "QR を読み取ってください"
        binding.detailText.text = "PC の画面に表示されている QR にカメラを向けてください"

        val scanner = BarcodeScanning.getClient()
        bindCamera(
            ImageAnalysis.Analyzer { image ->
                processBarcode(image, scanner)
            }
        )
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun processBarcode(proxy: ImageProxy, scanner: com.google.mlkit.vision.barcode.BarcodeScanner) {
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
        binding.statusText.text = getString(R.string.status_connecting)
        binding.detailText.text = "${target.role} として ${target.host}:${target.port} へ接続します"
        client.connect(target, deviceName())
    }

    private fun startStreaming() {
        mode = Mode.STREAMING
        poseAnalyzer?.close()

        val analyzer = PoseAnalyzer(this) { detection ->
            client.sendLandmarks(
                captureDeviceNanos = detection.captureDeviceNanos,
                width = detection.width,
                height = detection.height,
                landmarks = detection.landmarks,
            )
        }
        poseAnalyzer = analyzer
        bindCamera(analyzer)
        scheduleResync()
    }

    private fun stopStreaming() {
        mode = Mode.IDLE
        mainHandler.removeCallbacksAndMessages(null)
        client.disconnect()
        poseAnalyzer?.close()
        poseAnalyzer = null
        cameraSetup.stop()
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

    private fun bindCamera(analyzer: ImageAnalysis.Analyzer) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                cameraSetup.start(future.get(), analyzer) { info ->
                    runOnUiThread { binding.detailText.text = info }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    binding.statusText.text = getString(R.string.status_error)
                    binding.detailText.text = "カメラを開けませんでした: ${e.message}"
                }
            }
        }, ContextCompat.getMainExecutor(this))
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

        if (state == SensorClient.State.STREAMING && mode != Mode.STREAMING) {
            startStreaming()
        }
        if (state == SensorClient.State.ERROR) {
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
        // 毎フレーム更新すると UI スレッドを圧迫する。間引く。
        if (sent % PROGRESS_EVERY != 0L) return
        runOnUiThread {
            binding.detailText.text = "送信 $sent フレーム / 破棄 $dropped  " +
                String.format("往復 %.2f ms", timeSync.roundTripNanos / 1_000_000.0)
        }
    }

    companion object {
        private const val PROGRESS_EVERY = 30L
    }
}
