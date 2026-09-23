package com.murayama.wheelchairsensor.net

import java.security.MessageDigest

/** 生の Android ID をネットワークへ出さない。 */
object DeviceIdentity {
    fun hash(androidId: String): String = MessageDigest.getInstance("SHA-256")
        .digest(androidId.toByteArray(Charsets.UTF_8)).take(8)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
