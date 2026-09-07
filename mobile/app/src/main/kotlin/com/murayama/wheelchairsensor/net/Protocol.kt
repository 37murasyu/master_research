package com.murayama.wheelchairsensor.net

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

    /** 接続時の名乗り。 */
    fun hello(role: String, device: String, session: String): String =
        JSONObject().apply {
            put("type", "hello")
            put("v", VERSION)
            put("role", role)
            put("device", device)
            put("session", session)
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
