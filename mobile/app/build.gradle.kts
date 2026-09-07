import java.util.Properties

plugins {
    // Kotlin は AGP 9 に組み込まれているので、別プラグインは適用しない。
    id("com.android.application")
}

// リリース署名の設定。keystore.properties と鍵本体は git 管理外に置く。
//
// 鍵を固定するのは、後から更新版を配れるようにするため。Android は署名が
// 一致しない APK の上書きインストールを拒否するので、デバッグ鍵（PC ごとに
// 異なる）で配ると次の版が入らなくなる。
//
// 鍵が無い環境（CI など）でもビルドは通るようにし、その場合は署名なしにする。
val keystoreProperties = Properties().apply {
    val file = rootProject.file("keystore.properties")
    if (file.exists()) file.inputStream().use { load(it) }
}
val hasReleaseKey = keystoreProperties.getProperty("storeFile") != null &&
    rootProject.file(keystoreProperties.getProperty("storeFile", "")).exists()

android {
    namespace = "com.murayama.wheelchairsensor"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.murayama.wheelchairsensor"
        // MediaPipe Tasks の要求。Google 公式サンプルと同じ。
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // mediapipe のネイティブライブラリは 4 ABI で計 61MB ある。
        // 2019 年以降の Android 端末はすべて arm64-v8a で、Apple Silicon 上の
        // エミュレータも arm64。絞ることで APK が 79MB -> 約36MB になる。
        // 32bit 端末や Intel 機で動かす必要が出たらこの行を外すこと。
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    signingConfigs {
        if (hasReleaseKey) {
            create("release") {
                storeFile = rootProject.file(keystoreProperties.getProperty("storeFile"))
                storePassword = keystoreProperties.getProperty("storePassword")
                keyAlias = keystoreProperties.getProperty("keyAlias")
                keyPassword = keystoreProperties.getProperty("keyPassword")

                // v2 だけでもインストールはできるが、v3 は鍵のローテーションに
                // 対応する。鍵が漏れた場合に差し替える道を残しておく。
                enableV1Signing = false  // minSdk 24 なので不要。署名検証が速くなる
                enableV2Signing = true
                enableV3Signing = true
            }
        }
    }

    buildTypes {
        release {
            // ネイティブライブラリ（mediapipe）が大半なので、Kotlin 側を縮めても
            // ほとんど効かない。難読化で不具合を持ち込むリスクの方が大きい。
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")

            if (hasReleaseKey) {
                signingConfig = signingConfigs.getByName("release")
            } else {
                logger.warn(
                    "リリース鍵がありません（mobile/keystore.properties）。" +
                        "署名なしAPKになり、端末にインストールできません。"
                )
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    buildFeatures {
        viewBinding = true
    }

    // AGP の既定は src/main/java。Kotlin を kotlin/ に置いているので明示する。
    sourceSets {
        getByName("main") {
            kotlin.srcDirs("src/main/kotlin")
        }
        getByName("test") {
            kotlin.srcDirs("src/test/kotlin")
        }
    }

    androidResources {
        // .task モデルは既に圧縮済みのバイナリ。再圧縮しても縮まないうえ、
        // 実行時に展開コストがかかるので除外する。
        noCompress += "task"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

// 姿勢推定モデルはリポジトリルートの 1 つを正とし、ビルド時に assets へ複製する。
// 5.5MB のバイナリを 2 箇所で管理すると、更新したときに片方だけ古くなる。
// PC 側（app/core/resources.py 経由）と同じファイルを使うことがここで担保される。
val poseModel = rootProject.file("../pose_landmarker_lite.task")

val copyPoseModel by tasks.registering(Copy::class) {
    from(poseModel)
    into(layout.projectDirectory.dir("src/main/assets"))
    doFirst {
        require(poseModel.isFile) {
            "姿勢推定モデルが見つかりません: ${poseModel.absolutePath}\n" +
                "  リポジトリルートに pose_landmarker_lite.task を置いてください。"
        }
    }
}

tasks.named("preBuild") { dependsOn(copyPoseModel) }

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("com.google.android.material:material:1.12.0")

    // ライフサイクル（カメラの開始・停止をアクティビティに追従させる）
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    // CameraX。Camera2Interop はオートフォーカス／露出を固定するのに要る。
    // 固定しないと焦点距離＝カメラ内部パラメータが変わり、校正が無効になる。
    val cameraX = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // MediaPipe Tasks（Vision）。PC 側と同じ pose_landmarker_lite.task を使う。
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    // 接続先を QR で受け取る
    implementation("com.google.mlkit:barcode-scanning:17.3.0")

    // WebSocket
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // JSON（PC 側 app/net/protocol.py と同じ電文を組み立てる）
    implementation("org.json:json:20240303")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
