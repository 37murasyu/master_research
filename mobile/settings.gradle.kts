// 車椅子駆動計測 — Android センサアプリ
//
// スマホ側で MediaPipe を実行し、映像ではなく 33 点のランドマークだけを
// 時刻付きで PC へ送る。PC 側の受信層は app/net/ にある。
//
// バージョンは手元の別プロジェクトで動作実績のある組み合わせに合わせてある
// （AGP 9.1.0 / Gradle 9.3.1 / Java 17。Kotlin は AGP 組み込み）。

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    // AGP 9.0 以降は Kotlin サポートが組み込みになっており、
    // org.jetbrains.kotlin.android を別途適用するとエラーになる。
    id("com.android.application") version "9.1.0" apply false
}

rootProject.name = "WheelchairSensor"
include(":app")
