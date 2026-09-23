package com.murayama.wheelchairsensor.camera

import androidx.camera.core.ImageAnalysis

/**
 * カメラを何に使うか。解析フレームの形式と、光学系を固定するかが用途で違う。
 *
 * 同じ設定を使い回すと、QR 読み取りに計測用の RGBA が渡って ML Kit が落ち、
 * 焦点も無限遠のままで手元の QR が読めない。
 */
enum class CameraPurpose(
    val outputImageFormat: Int,
    val fixOptics: Boolean,
) {
    /**
     * PC の QR を読む。ML Kit の `InputImage.fromMediaImage` は JPEG と
     * YUV_420_888 しか受け付けない。数十 cm 先の画面を読むので焦点は自動にする。
     * 計測データは出さないので、校正の前提（焦点固定）は要らない。
     */
    SCAN_QR(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888, fixOptics = false),

    /** 姿勢推定。MediaPipe に Bitmap で渡すので RGBA。校正した K を保つため光学系を固定する。 */
    MEASURE(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888, fixOptics = true),
}
