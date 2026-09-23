package com.murayama.wheelchairsensor.net

import okio.ByteString.Companion.toByteString
import org.json.JSONArray
import org.json.JSONObject

/**
 * PC 側 `app/net/protocol.py` と対になる電文の定義。
 *
 * ここがずれると受信側が ProtocolError で黙って捨てるだけになるので、
 * フィールド名・型・単位を厳密に合わせる。特に:
 *
 * - `t_capture_ns` は **整数**。PC 側の `_require_int` は bool も含めて型を見る
 * - `lm` は **ちょうど 33 点**。数が違うと 1 フレームまるごと捨てられる
 * - `x`, `y` は **正規化座標 [0,1]**。ピクセル換算は PC 側が `w`,`h` を掛けて行う
 *   （既存の `utils.extract_keypoints` と同じ規約に揃えてある）
 */
object Protocol {

    const val VERSION = 1

    /** MediaPipe Pose のランドマーク数。pose_landmarker_lite.task の出力と一致する。 */
    const val LANDMARK_COUNT = 33

    const val ROLE_CAM0 = "cam0"
    const val ROLE_CAM1 = "cam1"

    val ROLES = listOf(ROLE_CAM0, ROLE_CAM1)

    fun isValidRole(role: String?): Boolean = role in ROLES

    /**
     * 接続時の名乗り。
     *
     * @param deviceId 端末ごとに変わらない識別子。同じ機種を 2 台使うと [device] はどちらも
     *   "Google Pixel 7a" になり、PC 側は校正時の端末と照合できない。null なら項目ごと送らない
     *   （PC 側は「省略可能な文字列」として読むので、null を送ると型違いで弾かれる）。
     */
    fun hello(role: String, device: String, session: String, deviceId: String? = null): String =
        JSONObject().apply {
            put("type", "hello")
            put("v", VERSION)
            put("role", role)
            put("device", device)
            put("session", session)
            if (deviceId != null) put("device_id", deviceId)
        }.toString()

    /** 時刻同期の要求。t1 は端末の単調時計。 */
    fun syncRequest(t1Nanos: Long): String =
        JSONObject().apply {
            put("type", "sync_req")
            put("t1", t1Nanos)
        }.toString()

    /**
     * ランドマーク 1 フレーム分。
     *
     * @param landmarks (x, y, z, visibility) が [LANDMARK_COUNT] 個。
     * @param captureNanosPcClock **PC 時計に補正済み**の撮影時刻。端末のローカル時計ではない。
     */
    fun landmarks(
        role: String,
        seq: Long,
        captureNanosPcClock: Long,
        width: Int,
        height: Int,
        landmarks: List<FloatArray>,
    ): String {
        require(landmarks.size == LANDMARK_COUNT) {
            "ランドマークは ${LANDMARK_COUNT} 点である必要があります（実際: ${landmarks.size}）"
        }

        val points = JSONArray()
        for (point in landmarks) {
            points.put(
                JSONArray().apply {
                    // 桁を落として電文を小さくする。1e-5 の精度があれば
                    // 1280px 幅で 0.013px 相当なので、丸めによる誤差は無視できる。
                    put(round5(point[0]))
                    put(round5(point[1]))
                    put(round5(point[2]))
                    put(round5(point[3]))
                }
            )
        }

        return JSONObject().apply {
            put("type", "landmarks")
            put("role", role)
            put("seq", seq)
            put("t_capture_ns", captureNanosPcClock)
            put("w", width)
            put("h", height)
            put("lm", points)
        }.toString()
    }

    private fun round5(value: Float): Double =
        Math.round(value.toDouble() * 100_000.0) / 100_000.0

    /**
     * 撮影要求への応答。姿勢推定に使っているのと同じフレームを JPEG にしたもの。
     *
     * base64 は標準の字母・パディングあり・改行なし。PC 側は `base64.b64decode(validate=True)`
     * で読むので、URL 用の字母や改行を混ぜると弾かれる。android.util.Base64 は JVM の
     * ユニットテストで動かず、java.util.Base64 は API 26 からなので、OkHttp が持つ okio を使う。
     *
     * @param captureNanosPcClock **PC 時計に補正済み**の撮影時刻。
     * @param width JPEG の幅（縮小した場合は縮小後の寸法）。
     */
    fun calibrationFrame(
        role: String,
        id: Long,
        captureNanosPcClock: Long,
        width: Int,
        height: Int,
        jpeg: ByteArray,
    ): String =
        JSONObject().apply {
            put("type", "calib_frame")
            put("role", role)
            put("id", id)
            put("t_capture_ns", captureNanosPcClock)
            put("w", width)
            put("h", height)
            put("jpeg", jpeg.toByteString().base64())
        }.toString()

    /** 受信した電文の種類。JSON として読めなければ null。 */
    fun messageType(raw: String): String? =
        try {
            JSONObject(raw).optString("type").ifEmpty { null }
        } catch (_: Exception) {
            null
        }

    /**
     * PC からの撮影要求。想定外・不正なら null（1 通で計測を止めない）。
     *
     * 値の範囲は PC 側 `protocol.py` の検査と同じにしてある。
     */
    fun parseCaptureRequest(raw: String): CaptureRequest? {
        return try {
            val json = JSONObject(raw)
            if (json.optString("type") != "capture_req" || !json.has("id")) return null
            val maxWidth = if (json.has("max_width")) json.getInt("max_width") else null
            val quality = if (json.has("quality")) json.getInt("quality") else null
            if (maxWidth != null && maxWidth <= 0) return null
            if (quality != null && quality !in 1..100) return null
            CaptureRequest(
                id = json.getLong("id"),
                atNanosPcClock = if (json.has("at_ns")) json.getLong("at_ns") else null,
                maxWidth = maxWidth,
                quality = quality,
            )
        } catch (_: Exception) {
            null
        }
    }

    /**
     * PC → 端末の撮影要求。
     *
     * @param atNanosPcClock PC 時計での目標撮影時刻。null なら次のフレーム。
     * @param maxWidth これより幅が大きければ縮めて返す（ライブ表示用）。null なら全解像度。
     * @param quality JPEG の画質。null なら端末の既定（校正用）。
     */
    data class CaptureRequest(
        val id: Long,
        val atNanosPcClock: Long?,
        val maxWidth: Int?,
        val quality: Int?,
    )

    /** PC からの応答。想定外の電文なら null。 */
    fun parseSyncResponse(raw: String): SyncResponse? {
        return try {
            val json = JSONObject(raw)
            if (json.optString("type") != "sync_res") return null
            SyncResponse(
                t1 = json.getLong("t1"),
                t2 = json.getLong("t2"),
                t3 = json.getLong("t3"),
            )
        } catch (_: Exception) {
            null
        }
    }

    data class SyncResponse(val t1: Long, val t2: Long, val t3: Long)
}
