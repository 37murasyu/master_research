package com.murayama.wheelchairsensor.capture

import android.graphics.Bitmap
import android.util.Log
import com.murayama.wheelchairsensor.net.Protocol
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

/**
 * 撮影要求に、解析中のフレームを JPEG にして応える。
 *
 * 解析のスレッドでは縮小（または複製）だけを行い、圧縮と送信は専用の [executor] に任せる。
 * 全解像度の JPEG 化は数十ミリ秒かかり、解析のスレッドで行うと推論のフレームが落ちる。
 *
 * 複製するのは、元の Bitmap を MediaPipe も読むため。別のスレッドで圧縮している間に
 * 元の Bitmap が回収されても影響を受けないようにする。
 *
 * 圧縮待ちが [maxInFlight] を超えたら新しい要求は捨てる。PC は応答が 1 秒来なければ
 * 次を出すので、溜めるより捨てるほうがよい。
 */
class JpegResponder(
    private val executor: ExecutorService,
    private val send: (
        request: Protocol.CaptureRequest,
        captureDeviceNanos: Long,
        width: Int,
        height: Int,
        jpeg: ByteArray,
    ) -> Unit,
    private val maxInFlight: Int = 2,
    private val defaultQuality: Int = 90,
) {

    private val inFlight = AtomicInteger(0)

    /** 圧縮待ちが詰まっていて捨てた要求の数。 */
    val dropped = AtomicLong(0)

    /** 解析のスレッドから呼ぶ。[bitmap] は正立済みの RGBA。 */
    fun offer(bitmap: Bitmap, captureDeviceNanos: Long, requests: List<Protocol.CaptureRequest>) {
        for (request in requests) {
            if (inFlight.get() >= maxInFlight) {
                dropped.incrementAndGet()
                continue
            }
            val source = copyFor(bitmap, request.maxWidth) ?: continue
            inFlight.incrementAndGet()
            try {
                executor.execute { encodeAndSend(source, request, captureDeviceNanos) }
            } catch (_: RejectedExecutionException) {
                // 画面を閉じて executor が止まった後
                inFlight.decrementAndGet()
                source.recycle()
            }
        }
    }

    private fun encodeAndSend(source: Bitmap, request: Protocol.CaptureRequest, captureDeviceNanos: Long) {
        try {
            val out = ByteArrayOutputStream()
            source.compress(Bitmap.CompressFormat.JPEG, request.quality ?: defaultQuality, out)
            send(request, captureDeviceNanos, source.width, source.height, out.toByteArray())
        } catch (e: Exception) {
            Log.w(TAG, "撮影要求 ${request.id} に応えられませんでした", e)
        } finally {
            source.recycle()
            inFlight.decrementAndGet()
        }
    }

    /** 幅が [maxWidth] を超えていれば縮め、そうでなければ複製する。 */
    private fun copyFor(bitmap: Bitmap, maxWidth: Int?): Bitmap? {
        if (maxWidth != null && maxWidth < bitmap.width) {
            val height = (bitmap.height.toDouble() * maxWidth / bitmap.width).roundToInt().coerceAtLeast(1)
            return Bitmap.createScaledBitmap(bitmap, maxWidth, height, true)
        }
        return bitmap.copy(Bitmap.Config.ARGB_8888, false)
    }

    companion object {
        private const val TAG = "JpegResponder"
    }
}
