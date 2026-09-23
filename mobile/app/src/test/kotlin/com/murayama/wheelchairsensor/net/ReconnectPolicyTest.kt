package com.murayama.wheelchairsensor.net

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 切れたときに自動でつなぎ直すかどうか。
 *
 * PC がはっきり断った（役割違い・別の session・校正と違う端末＝1008、別の接続に席を
 * 譲った＝4000）ときにつなぎ直すと、断られ続けて画面の案内も消える。
 */
class ReconnectPolicyTest {

    @Test
    fun `PC 側が止まった・通信が切れたときはつなぎ直す`() {
        assertTrue(SensorClient.isRetryableClose(1000))
        assertTrue(SensorClient.isRetryableClose(1001))  // PC 側のツールを終えた
        assertTrue(SensorClient.isRetryableClose(1006))  // 異常切断
        assertTrue(SensorClient.isRetryableClose(1011))  // PC 側の内部エラー
    }

    @Test
    fun `PC がはっきり断ったときはつなぎ直さない`() {
        assertFalse(SensorClient.isRetryableClose(1008))
        assertFalse(SensorClient.isRetryableClose(4000))
    }
}
