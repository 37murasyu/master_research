package com.murayama.wheelchairsensor.pose

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * カメラのフレームから姿勢ランドマークを取り出す。
 *
 * PC 側と**同じモデル**（pose_landmarker_lite.task）を使う。研究リポジトリに
 * 置いてあるものをそのまま assets に入れてあるので、両者でランドマークの
 * 定義や精度が食い違わない。
 *
 * 映像そのものは送らない。33 点のランドマークだけなら 1 フレーム約 600 バイトで、
 * 30fps でも 18 KB/s に収まる。PC 側は映像のデコードが不要になる。
 */
class PoseAnalyzer(
    context: Context,
    private val onResult: (Detection) -> Unit,
) : ImageAnalysis.Analyzer {

    /** 1 フレーム分の結果。時刻は端末の単調時計。 */
    data class Detection(
        val captureDeviceNanos: Long,
        val width: Int,
        val height: Int,
        val landmarks: List<FloatArray>,
    )

    private var landmarker: PoseLandmarker? = null

    /** 送り出したフレームの情報。結果が返ってきたときに突き合わせる。 */
    private data class PendingFrame(val captureNanos: Long, val width: Int, val height: Int)

    /**
     * MediaPipe に渡した時刻（ミリ秒）から元のフレーム情報へ戻すための対応表。
     * LIVE_STREAM モードは結果が非同期で返るため、どのフレームの結果かを
     * タイムスタンプで突き合わせる必要がある。
     *
     * **サイズも一緒に持つ**のが要点。1 個の変数で「直近のサイズ」を持つと、
     * フレーム N の結果にフレーム N+k のサイズを付けてしまう。PC 側は
     * 正規化座標に w/h を掛けてピクセルに直すので、取り違えると
     * 3D 再構成が静かに狂う。
     */
    private val pendingFrames = HashMap<Long, PendingFrame>()

    init {
        try {
            val base = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setResultListener { result, _ -> handleResult(result) }
                .setErrorListener { error -> Log.e(TAG, "推論に失敗しました", error) }
                .build()

            landmarker = PoseLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "PoseLandmarker を初期化できませんでした", e)
        }
    }

    override fun analyze(image: ImageProxy) {
        val detector = landmarker
        if (detector == null) {
            image.close()
            return
        }

        try {
            val captureNanos = SystemClock.elapsedRealtimeNanos()
            val bitmap = image.toUprightBitmap()
            val timestampMs = captureNanos / 1_000_000

            synchronized(pendingFrames) {
                pendingFrames[timestampMs] =
                    PendingFrame(captureNanos, bitmap.width, bitmap.height)
                // 結果が返らなかったフレームの記録が溜まらないようにする
                if (pendingFrames.size > PENDING_LIMIT) {
                    val oldest = pendingFrames.keys.minOrNull()
                    if (oldest != null) pendingFrames.remove(oldest)
                }
            }

            detector.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
        } catch (e: Exception) {
            Log.w(TAG, "フレームを処理できませんでした", e)
        } finally {
            // 必ず閉じる。閉じ忘れるとカメラのバッファが尽きて映像が止まる。
            image.close()
        }
    }

    private fun handleResult(result: PoseLandmarkerResult) {
        val poses = result.landmarks()
        if (poses.isEmpty()) return  // 人が写っていないフレームは送らない

        val timestampMs = result.timestampMs()
        // 対応するフレームが見つからない結果は捨てる。サイズを推測して
        // 送ると、PC 側で誤ったピクセル座標になる。
        val pending = synchronized(pendingFrames) {
            pendingFrames.remove(timestampMs)
        } ?: return

        val points = poses[0].map { landmark ->
            floatArrayOf(
                landmark.x(),
                landmark.y(),
                landmark.z(),
                landmark.visibility().orElse(1.0f),
            )
        }

        onResult(
            Detection(
                captureDeviceNanos = pending.captureNanos,
                width = pending.width,
                height = pending.height,
                landmarks = points,
            )
        )
    }

    /**
     * ImageProxy を、画面表示と同じ向きの Bitmap にする。
     *
     * 回転を戻しておかないと、正規化座標の x/y が横倒しのままになり、
     * PC 側で `x * w`, `y * h` としたときに軸が入れ替わる。
     */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val source = toBitmap()
        val degrees = imageInfo.rotationDegrees
        if (degrees == 0) return source
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    fun close() {
        landmarker?.close()
        landmarker = null
    }

    companion object {
        private const val TAG = "PoseAnalyzer"

        /** PC 側と同じモデル。研究リポジトリのものを assets に入れてある。 */
        const val MODEL_ASSET = "pose_landmarker_lite.task"

        private const val PENDING_LIMIT = 64
    }
}
