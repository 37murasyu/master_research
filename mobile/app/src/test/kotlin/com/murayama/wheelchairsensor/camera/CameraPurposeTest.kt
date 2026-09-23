package com.murayama.wheelchairsensor.camera

import androidx.camera.core.ImageAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 用途ごとのカメラ設定を固定する。
 *
 * QR 読み取りが計測用の設定（RGBA・焦点固定）を借りていたため、ML Kit が
 * 「Only JPEG and YUV_420_888 are supported now」で落ちていた（Pixel 7a で再現）。
 */
class CameraPurposeTest {

    @Test
    fun `QR 読み取りは ML Kit が受け付ける YUV_420_888 で出す`() {
        assertEquals(
            ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888,
            CameraPurpose.SCAN_QR.outputImageFormat,
        )
    }

    @Test
    fun `QR 読み取りは焦点を固定しない`() {
        // 無限遠に固定すると、手元の PC 画面の QR がぼけて読めない。
        assertFalse(CameraPurpose.SCAN_QR.fixOptics)
    }

    @Test
    fun `計測は MediaPipe に渡す RGBA で出し、校正の前提どおり光学系を固定する`() {
        assertEquals(
            ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888,
            CameraPurpose.MEASURE.outputImageFormat,
        )
        assertTrue(CameraPurpose.MEASURE.fixOptics)
    }
}
