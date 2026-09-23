package com.murayama.wheelchairsensor.camera

import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit

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
    private var camera: Camera? = null
    private var purpose: CameraPurpose? = null

    /**
     * [pinFrameRate] が真なら（光学系を固定する用途のとき）、自動露出の目標フレームレートを 30 fps に固定する。
     * 暗い室内では自動露出が露光を 1/15 s ほどに延ばし、カメラ自体が 15 fps に落ちる（実機の Pixel 7a で PC に
     * 10〜15 Hz しか届かなかった原因の候補）。固定すると映像は暗くなる。
     * [onSensorFrame] はカメラが 1 枚撮るたびに呼ぶ（Camera2 の撮影完了。解析に渡らなかったフレームも数える）。
     */
    fun start(
        provider: ProcessCameraProvider,
        purpose: CameraPurpose,
        analyzer: ImageAnalysis.Analyzer,
        pinFrameRate: Boolean = false,
        onSensorFrame: (() -> Unit)? = null,
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

        if (purpose.fixOptics) applyFixedOptics(analysisBuilder, pinFrameRate)
        if (onSensorFrame != null) {
            Camera2Interop.Extender(analysisBuilder).setSessionCaptureCallback(
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        session: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult,
                    ) = onSensorFrame()
                }
            )
        }

        val analysis = analysisBuilder.build().also {
            it.setAnalyzer(analysisExecutor, analyzer)
        }

        camera = provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis,
        )
        this.purpose = purpose
        // QR 読み取りは最初に画面の中央へピントを合わせる。既定の連続 AF だけだと、
        // 画面に映した QR のような細かい模様で微妙に外れたまま落ち着くことがある。
        if (!purpose.fixOptics) previewView.post { focusCenter() }

        val optics = when {
            purpose.fixOptics && pinFrameRate -> "AF・AE・AWB 固定、${TARGET_FPS} fps 固定"
            purpose.fixOptics -> "AF・AE・AWB 固定"
            else -> "オートフォーカス"
        }
        // 実際の解像度は目標（TARGET_RESOLUTION）と違うことがある（4:3 が選ばれる）。表示は実際のフレームから出す
        onReady(optics)
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
    private fun applyFixedOptics(builder: ImageAnalysis.Builder, pinFrameRate: Boolean) {
        try {
            if (pinFrameRate) {
                // 露光を 1/30 s 以内に抑え、カメラが 30 fps を出すようにする（AE のロックはこの範囲で収束した値を固める）
                Camera2Interop.Extender(builder).setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(TARGET_FPS, TARGET_FPS),
                )
            }
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
                // 光学式の手ブレ補正はレンズや撮像素子を動かし、光学中心（K の cx, cy）が
                // フレームごとにずれる。電子式だけ切っても残るので、こちらも切る。
                .setCaptureRequestOption(
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
                )
        } catch (e: Exception) {
            Log.w(TAG, "光学系の固定に一部失敗しました。精度が落ちる可能性があります。", e)
        }
    }

    /**
     * プレビュー上の点 ([x], [y]) にピントと露出を合わせる（タップでのピント合わせ）。
     *
     * **光学系を固定している用途（計測）では何もしない**。レンズが動くと、校正で求めた
     * 内部パラメータが実際の映像と合わなくなる。3 秒後に連続 AF へ戻る。
     */
    fun focusAt(x: Float, y: Float) {
        val current = camera ?: return
        if (purpose?.fixOptics != false) return
        val point = previewView.meteringPointFactory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(
            point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
        )
            .setAutoCancelDuration(3, TimeUnit.SECONDS)
            .build()
        current.cameraControl.startFocusAndMetering(action)
    }

    /** 画面の中央にピントを合わせる。QR は中央に写すことが多い。 */
    fun focusCenter() = focusAt(previewView.width / 2f, previewView.height / 2f)

    fun stop() {
        provider?.unbindAll()
        provider = null
        camera = null
        purpose = null
    }

    companion object {
        private const val TAG = "CameraSetup"

        /** PC 側の既存パイプラインが 1280x720 前提（config.frame_shape）。 */
        val TARGET_RESOLUTION = Size(1280, 720)

        /** 30fps 固定のときのフレームレート。PC 側の格子（同期バッファ）と同じ */
        const val TARGET_FPS = 30
    }
}
