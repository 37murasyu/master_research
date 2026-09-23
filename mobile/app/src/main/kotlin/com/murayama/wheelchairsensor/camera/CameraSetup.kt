package com.murayama.wheelchairsensor.camera

import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executor

/**
 * カメラを開き、姿勢推定にフレームを流す。
 *
 * **オートフォーカスと自動露出を固定するのが最も重要**。オートフォーカスが
 * 動くとレンズの焦点距離が変わり、キャリブレーションで求めた内部パラメータ
 * （カメラ行列 K）が実態と合わなくなる。三角測量はその K を前提にしているので、
 * 校正した意味が消える。撮影中に明るさが変わるのも、姿勢推定の安定性を損なう。
 *
 * CameraX の標準 API には焦点固定の口が無いので Camera2Interop を使う。
 */
class CameraSetup(
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val analysisExecutor: Executor,
) {

    private var provider: ProcessCameraProvider? = null

    fun start(
        provider: ProcessCameraProvider,
        purpose: CameraPurpose,
        analyzer: ImageAnalysis.Analyzer,
        onReady: (String) -> Unit,
    ) {
        this.provider = provider
        provider.unbindAll()

        val resolution = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(TARGET_RESOLUTION, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
            )
            .build()

        val preview = Preview.Builder()
            .setResolutionSelector(resolution)
            .build()
            .also { it.surfaceProvider = previewView.surfaceProvider }

        val analysisBuilder = ImageAnalysis.Builder()
            .setResolutionSelector(resolution)
            // 溜まったフレームを処理しても意味がない。PC 側は時刻でペアを組むので、
            // 遅れて届いたフレームは使われない。常に最新だけを見る。
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(purpose.outputImageFormat)
            // CameraX 側で正立させる。Java 側で回すと 1280x720 の Bitmap を
            // 毎フレーム 2 枚確保して 1 枚捨てることになり、30fps で約 210MB/s の
            // アロケーションになる。
            .setOutputImageRotationEnabled(true)

        if (purpose.fixOptics) applyFixedOptics(analysisBuilder)

        val analysis = analysisBuilder.build().also {
            it.setAnalyzer(analysisExecutor, analyzer)
        }

        provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis,
        )

        val optics = if (purpose.fixOptics) "AF・AE・AWB 固定" else "オートフォーカス"
        onReady("解像度 ${TARGET_RESOLUTION.width}x${TARGET_RESOLUTION.height} / $optics")
    }

    /**
     * フォーカス・露出・ホワイトバランスを固定する。
     *
     * - AF: モードを OFF にし、レンズ位置を無限遠に固定する。**これが校正の前提**。
     * - AE / AWB: ロックをかけ、明るさや色温度が撮影中に変わらないようにする。
     *
     * 機種によっては一部が効かない。効かなくても撮影自体は続けられるので、
     * 例外にはせず警告を出すに留める（ただし精度は落ちる）。
     */
    private fun applyFixedOptics(builder: ImageAnalysis.Builder) {
        try {
            Camera2Interop.Extender(builder)
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_OFF,
                )
                // 0.0 は無限遠。被写体まで数メートルの想定なので実用上これでよい。
                // 近距離で撮るなら機種ごとに調整が要る。
                .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, 0.0f)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, true)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, true)
                // 手ブレ補正は画像を電子的に歪ませるため、校正した内部パラメータと
                // 合わなくなる。切っておく。
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
                )
        } catch (e: Exception) {
            Log.w(TAG, "光学系の固定に一部失敗しました。精度が落ちる可能性があります。", e)
        }
    }

    fun stop() {
        provider?.unbindAll()
        provider = null
    }

    companion object {
        private const val TAG = "CameraSetup"

        /** PC 側の既存パイプラインが 1280x720 前提（config.frame_shape）。 */
        val TARGET_RESOLUTION = Size(1280, 720)
    }
}
