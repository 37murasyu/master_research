package com.murayama.wheelchairsensor.net

import java.io.File
import org.json.JSONArray
import org.junit.Test

/**
 * PC 側と突き合わせるための「実際に送る電文」を書き出す。
 *
 * Kotlin が組み立てた生の JSON をファイルに落とし、Python 側のテスト
 * （tests/test_protocol_contract.py）がそれを decode できることを確かめる。
 * 片側だけのテストでは「両者が同じものを想定している」ことは保証できない。
 *
 * ここを更新したら `./gradlew :app:testDebugUnitTest` を実行して
 * ファイルを作り直し、Python 側のテストも通ることを確認すること。
 */
class ContractSampleTest {

    @Test
    fun `実際に送る電文を書き出す`() {
        val landmarks = List(Protocol.LANDMARK_COUNT) { i ->
            floatArrayOf(
                0.1f + i * 0.01f,
                0.2f + i * 0.005f,
                -0.05f + i * 0.002f,
                0.95f,
            )
        }

        val samples = JSONArray().apply {
            put(Protocol.hello("cam0", "Pixel 8", "3fc590ba"))
            put(Protocol.syncRequest(987_654_321_000L))
            put(
                Protocol.landmarks(
                    role = "cam0",
                    seq = 42L,
                    captureNanosPcClock = 1_725_699_123_456_789_000L,
                    width = 1280,
                    height = 720,
                    landmarks = landmarks,
                )
            )
            put(
                Protocol.landmarks(
                    role = "cam1",
                    seq = 43L,
                    captureNanosPcClock = 1_725_699_123_490_123_000L,
                    width = 1920,
                    height = 1080,
                    landmarks = landmarks,
                )
            )
        }

        val out = File(OUTPUT_PATH)
        out.parentFile?.mkdirs()
        out.writeText(samples.toString(2), Charsets.UTF_8)
    }

    companion object {
        /** テストの作業ディレクトリは app/ なので、そこからの相対パス。 */
        private const val OUTPUT_PATH = "../contract/golden_messages.json"
    }
}
