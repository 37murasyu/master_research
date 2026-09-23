package com.murayama.wheelchairsensor.net

import java.io.File
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * PC → 端末の電文を、PC 側が書き出した見本（mobile/contract/pc_messages.json）で確かめる。
 *
 * 見本は Python のテスト（tests/test_protocol_contract.py）が作る。PC 側で電文の形を
 * 変えたら `UPDATE_CONTRACT=1` で作り直し、このテストも通ることを確認すること。
 * 逆向き（端末 → PC）は ContractSampleTest が受け持つ。
 */
class PcMessagesContractTest {

    private fun samples(): List<String> {
        val file = File(INPUT_PATH)
        assertTrue("PC 側の見本がありません: ${file.absolutePath}", file.isFile)
        val array = JSONArray(file.readText(Charsets.UTF_8))
        return List(array.length()) { array.getString(it) }
    }

    @Test
    fun `PC が送る電文はすべて読み取れる`() {
        for (raw in samples()) {
            when (Protocol.messageType(raw)) {
                "sync_res" -> assertTrue(raw, Protocol.parseSyncResponse(raw) != null)
                "capture_req" -> assertTrue(raw, Protocol.parseCaptureRequest(raw) != null)
                else -> throw AssertionError("端末が知らない電文です: $raw")
            }
        }
    }

    @Test
    fun `ライブ表示用の撮影要求は縮小と画質の指定を運ぶ`() {
        val preview = samples()
            .filter { Protocol.messageType(it) == "capture_req" }
            .mapNotNull { Protocol.parseCaptureRequest(it) }
            .first { it.maxWidth != null }
        assertEquals(640, preview.maxWidth)
        assertEquals(70, preview.quality)
    }

    companion object {
        /** テストの作業ディレクトリは app/ なので、そこからの相対パス。 */
        private const val INPUT_PATH = "../contract/pc_messages.json"
    }
}
