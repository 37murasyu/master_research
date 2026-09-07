plugins {
    // Kotlin は AGP 9 に組み込まれているので、別プラグインは適用しない。
    id("com.android.application")
}

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

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
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
