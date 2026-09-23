package com.murayama.wheelchairsensor.net
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
class DeviceIdentityTest {
    @Test fun `識別子は SHA256 の先頭8バイトで安定する`() {
        assertEquals("ba7816bf8f01cfea", DeviceIdentity.hash("abc"))
        assertNotEquals(DeviceIdentity.hash("abc"), DeviceIdentity.hash("abcd"))
    }
}
