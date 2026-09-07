// ルートは空でよい。設定は settings.gradle.kts と app/build.gradle.kts に置く。
tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
