package com.murayama.wheelchairsensor.net

import android.net.Uri

/**
 * PC が QR で配る接続先。
 *
 *     ws://192.168.1.17:8765/?session=3fc590ba&role=cam0
 *
 * PC 側は `LandmarkServer.connect_url()` がこの形を出す。
 * role が cam0 / cam1 のどちらかを QR 自体が持つので、端末側で選ばせる必要がない
 * （選ばせると 2 台とも同じ役割にしてしまう事故が起きやすい）。
 */
data class ConnectionTarget(
    val url: String,
    val role: String,
    val session: String,
) {
    val host: String get() = Uri.parse(url).host ?: ""
    val port: Int get() = Uri.parse(url).port

    companion object {

        /**
         * QR の中身を解釈する。形が違えば理由つきで失敗を返す。
         *
         * 黙って null を返さないのは、現場で「読み取っても繋がらない」ときに
         * 何が悪いのか分からないと詰むため。
         */
        fun parse(raw: String): Result<ConnectionTarget> {
            val text = raw.trim()
            if (text.isEmpty()) {
                return Result.failure(IllegalArgumentException("QR の内容が空です"))
            }

            val uri = try {
                Uri.parse(text)
            } catch (e: Exception) {
                return Result.failure(IllegalArgumentException("URL として読めません: $text"))
            }

            val scheme = uri.scheme?.lowercase()
            if (scheme != "ws" && scheme != "wss") {
                return Result.failure(
                    IllegalArgumentException("ws:// で始まる必要があります（読み取った値: $text）")
                )
            }
            if (uri.host.isNullOrBlank()) {
                return Result.failure(IllegalArgumentException("接続先のホストがありません: $text"))
            }

            val role = uri.getQueryParameter("role")
            if (!Protocol.isValidRole(role)) {
                return Result.failure(
                    IllegalArgumentException(
                        "role が不正です（${role ?: "指定なし"}）。cam0 か cam1 である必要があります"
                    )
                )
            }

            val session = uri.getQueryParameter("session").orEmpty()

            return Result.success(
                ConnectionTarget(url = text, role = role!!, session = session)
            )
        }
    }
}
