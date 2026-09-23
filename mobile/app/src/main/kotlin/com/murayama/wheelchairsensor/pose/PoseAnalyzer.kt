package com.murayama.wheelchairsensor.pose

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
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
 *
 * 例外は撮影要求（校正とライブ表示）。[captureSink] には、推論に渡す直前のフレームを
 * **人が写っているかに関係なく**毎フレーム渡す（チェッカーボードだけを写す場面があるため）。
 * 渡された側は、応えるべき要求があるときだけ手元に写しを取る。
 */
class PoseAnalyzer(
    context: Context,
    private val captureSink: ((bitmap: Bitmap, captureDeviceNanos: Long) -> Unit)? = null,
    /** 段階ごとの数（解析・推論・人）。計測中の画面に 1 秒あたりの数を出して、遅い段を見分ける */
    private val counter: StageCounter? = null,
    /** GPU で推論する。使えなければ CPU に戻す（[delegateName] で分かる） */
    useGpu: Boolean = false,
    private val onResult: (Detection) -> Unit,
) : ImageAnalysis.Analyzer {

    // PoseLandmarker が Context を保持する場合に Activity 全体（PreviewView や
    // フレームバッファを含む）を道連れにしないよう、application の方を使う。
    private val appContext: Context = context.applicationContext

    /** 1 フレーム分の結果。時刻は端末の単調時計。 */
    data class Detection(
        val captureDeviceNanos: Long,
        val width: Int,
        val height: Int,
        val landmarks: List<FloatArray>,
    )

    @Volatile
    private var landmarker: PoseLandmarker? = null

    /** 実際に推論に使っているもの（"CPU" / "GPU"）。GPU を頼んでも初期化に失敗すれば CPU */
    var delegateName: String = "CPU"
        private set

    /** 推定器を閉じるのと、推定器へ画像を渡すのを排他にする（analyze / close）。 */
    private val lifecycleLock = Any()

    /** 実際に解析しているフレームの寸法。CameraX は目標の 1280x720 ではなく 4:3 を選ぶことがある。 */
    @Volatile
    var lastFrameWidth = 0
        private set

    @Volatile
    var lastFrameHeight = 0
        private set

    /** 最後に人を検出した時刻（端末の単調時計）。画面に「人が写っていない」を出すのに使う。 */
    @Volatile
    var lastPersonNanos = 0L
        private set

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
    private val pendingFrames = object : LinkedHashMap<Long, PendingFrame>(32, 0.75f, false) {
        // 溢れたら最古を 1 件落とす。以前は毎回 keys.minOrNull() で全体を走査しており、
        // 一度上限に達すると毎フレーム O(n) の探索がロックの中で走っていた。
        override fun removeEldestEntry(eldest: Map.Entry<Long, PendingFrame>): Boolean =
            size > PENDING_LIMIT
    }

    init {
        val delegates = if (useGpu) listOf(Delegate.GPU, Delegate.CPU) else listOf(Delegate.CPU)
        for (delegate in delegates) {
            landmarker = create(delegate)
            if (landmarker != null) {
                delegateName = delegate.name
                break
            }
        }
    }

    private fun create(delegate: Delegate): PoseLandmarker? {
        return try {
            val base = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(delegate)
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

            PoseLandmarker.createFromOptions(appContext, options)
        } catch (e: Exception) {
            Log.e(TAG, "PoseLandmarker を ${delegate.name} で初期化できませんでした", e)
            null
        }
    }

    override fun analyze(image: ImageProxy) {
        if (landmarker == null) {
            image.close()
            return
        }

        counter?.mark(Stage.ANALYZED)
        try {
            val captureNanos = SystemClock.elapsedRealtimeNanos()
            // CameraSetup が setOutputImageRotationEnabled(true) を指定しているので、
            // ここに届く時点で既に正立している。
            val bitmap = image.toBitmap()
            val timestampMs = captureNanos / 1_000_000

            // 推論と同じフレームで撮影要求に応える。失敗しても推論は続ける
            captureSink?.let { sink ->
                try {
                    sink(bitmap, captureNanos)
                } catch (e: Exception) {
                    Log.w(TAG, "撮影要求を処理できませんでした", e)
                }
            }

            synchronized(pendingFrames) {
                pendingFrames[timestampMs] =
                    PendingFrame(captureNanos, bitmap.width, bitmap.height)
            }
            lastFrameWidth = bitmap.width
            lastFrameHeight = bitmap.height

            // close() と同じロックの中で渡す。閉じかけの推定器に渡すと、ネイティブ側が
            // 解放済みのメモリに触れてプロセスごと落ちる（detectAsync の画像生成で
            // SIGSEGV / SIGABRT になった記録がある）。
            synchronized(lifecycleLock) {
                val current = landmarker ?: return
                current.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
            }
        } catch (e: Exception) {
            Log.w(TAG, "フレームを処理できませんでした", e)
        } finally {
            // 必ず閉じる。閉じ忘れるとカメラのバッファが尽きて映像が止まる。
            image.close()
        }
    }

    private fun handleResult(result: PoseLandmarkerResult) {
        // 対応するフレームが見つからない結果は捨てる。サイズを推測して
        // 送ると、PC 側で誤ったピクセル座標になる。
        //
        // 姿勢が空でも先に remove するのが要点。後回しにすると、人が写って
        // いないフレームの記録が永久に残り、接続直後の数秒で上限に達する。
        val pending = synchronized(pendingFrames) {
            pendingFrames.remove(result.timestampMs())
        } ?: return
        counter?.mark(Stage.INFERRED)
        // フレームを受け取ってから結果が返るまで（Bitmap への変換を含む）
        counter?.recordLatency(SystemClock.elapsedRealtimeNanos() - pending.captureNanos)

        val poses = result.landmarks()
        if (poses.isEmpty()) return  // 人が写っていないフレームは送らない
        counter?.mark(Stage.PERSON)
        lastPersonNanos = SystemClock.elapsedRealtimeNanos()

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
     * 推定器を閉じる。解析のスレッドが detectAsync の途中なら、それが終わるのを待ってから閉じる。
     */
    fun close() {
        val closing = synchronized(lifecycleLock) {
            landmarker.also { landmarker = null }
        }
        closing?.close()
    }

    companion object {
        private const val TAG = "PoseAnalyzer"

        /** PC 側と同じモデル。研究リポジトリのものを assets に入れてある。 */
        const val MODEL_ASSET = "pose_landmarker_lite.task"

        private const val PENDING_LIMIT = 64
    }
}
