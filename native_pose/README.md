# native_pose: MediaPipeをC++でDLL化する選択肢と手順

このフォルダは、Python から呼び出す C++ DLL（`native_pose.dll`）用です。現在の実装は「スタブ（検出なし）」ですが、ここを本物の C++ 推論に置き換えることで MediaPipe 関連の処理を C++ 化できます。

本書では、Windows での現実的な実装ルートと注意点を整理します。

## 目的と前提

- 目的: Python 側の MediaPipe Tasks 依存を減らし、推論部分を C++ DLL に集約する。
- 期待効果:
  - Python ↔ C++ のブリッジコスト削減（数 ms 程度）
  - C++ 側で `num_threads` 等の微調整が可能（モデル/ランタイム次第）
  - ランタイム選択（XNNPACK/DirectML/ONNX Runtime など）で CPU/GPU 最適化を検討できる
- 非現実的な期待:
  - 「Python をやめるだけ」で推論時間が劇的に速くなること。実際には時間の大半はモデル推論そのものに費やされ、言語境界のコストはごく小さいです。

## 選択肢の比較（概要）

1. MediaPipe Tasks C++ を直接使う（推奨だが Windows では重め）

  - 内容: `mediapipe/tasks/vision/pose_landmarker` の C++ API を DLL から呼ぶ
  - 長所: 既存 .task モデルをそのまま利用、後処理（ワールド座標/可視度など）も再利用
  - 短所: Windows でのビルドは Bazel 依存で重量級。abseil/flatbuffers/tflite などの依存解決が必要
  - 実装方針（例）:
    1. Bazelisk + Visual Studio Build Tools を用意
    2. mediapipe を clone（サブモジュール/依存の取得）
    3. pose landmarker の静的ライブラリ or DLL を Bazel でビルド
    4. 本 DLL とリンクするラッパ（本フォルダの CMake プロジェクト）を作成
    5. `NPOSE_WITH_MEDIAPIPE` を有効化して `npose_create/detect` 内で Tasks API を呼ぶ

2. TensorFlow Lite C++ を直接叩く（BlazePose/MoveNet 等の TFLite モデル）

  - 内容: TFLite C++ API + XNNPACK マルチスレッドで推論。後処理は自前実装
  - 長所: 依存が比較的軽い。`SetNumThreads()` など細かな制御がしやすい
  - 短所: MediaPipe の後処理（ランドマーク正規化、ワールド座標推定、スムージング等）を再実装する必要

3. ONNX Runtime で代替モデル（MoveNet/BlazePose ONNX 等）

  - 内容: ONNX Runtime C++ API で高速推論。Windows で導入が最も簡単
  - 長所: 公式の prebuilt が豊富、CMake 連携が容易、DirectML や CUDA のバックエンドも選べる
  - 短所: モデル/キーポイント仕様が MediaPipe と異なる場合が多い（33 点→17 点など）。既存処理との整合が必要

実務的には「まず 3) で統合・性能確認 → 1) に挑戦」が現実的な順序です。Python Tasks と同等の 33 点が必要であれば 1) が王道です。

## 既存 ABI とデータ契約

- C 関数:
  - `npose_create(const char* model_path, int num_threads, npose_handle_t* out)`
  - `npose_detect(npose_handle_t, const uint8_t* bgra, int w, int h, int stride, npose_result_t* out)`
  - `npose_destroy(npose_handle_t)`
- 画像入力: BGRA, row-major, `stride = width * 4`
- 出力: `npose_result_t`（正規化座標 x,y∈[0,1], z 任意, visibility∈[0,1]）。MediaPipe 互換を想定

## 実装の道筋（詳細）

### A. MediaPipe Tasks C++ 版（.task をそのまま利用）

1. 環境準備
  - Bazelisk（Windows 対応）
  - Visual Studio 2022 C++ 開発ツール
  - Git, Python（Bazel のスクリプトで必要になることが多い）
2. mediapipe リポジトリ取得
  - `git clone https://github.com/google/mediapipe`
  - `cd mediapipe`
3. ビルドターゲットの確認
  - `mediapipe/tasks/vision/pose_landmarker` の C++ ターゲットを DLL/静的ライブラリにビルド
  - Bazel のカスタム BUILD でエクスポート可能な形にまとめる必要あり
4. 本 DLL との連携
  - Bazel 生成物（.lib/.dll/.headers）を `third_party/mediapipe/` 的な場所に配置
  - 本プロジェクトの CMake で `NPOSE_WITH_MEDIAPIPE=ON` を有効にし、`target_link_libraries` でリンク
  - `pose_landmarker.cpp` 内の `#ifdef NPOSE_WITH_MEDIAPIPE` 実装を有効化

注意: 公式に Windows 向け prebuilt が揃っていないため、メンテ負荷が最も高い選択肢です。

### B. TFLite 直叩き
- TFLite C API or C++ API を vcpkg 等で導入
- BlazePose 系モデル（.tflite）を読み込み、入出力テンソル前後処理を C++ 実装
- XNNPACK を有効化し `SetNumThreads(num_threads)` を設定
- MediaPipe の後処理を必要分だけ再現

### C. ONNX Runtime 版（導入容易）
- ONNX Runtime の prebuilt をダウンロードし、`third_party/onnxruntime/` へ配置
- CMake で `onnxruntime.lib` をリンク、`NPOSE_WITH_ONNXRUNTIME` を定義
- `pose_landmarker.cpp` の `#ifdef` ブロックで ONNX 実装を有効化
- モデルは MoveNet/BlazePose ONNX を利用（キーポイント数の差に注意）

## パフォーマンスに関する現実的な見積もり
- Python → C++ 移行だけで得られるのは「数 ms」の改善が多い
- 真のボトルネックは「モデル推論」
  - 改善策: より軽いモデル、入力解像度の最適化、フレーム間サンプリング、マルチスレッド、GPU/DirectML/CUDA バックエンド
- したがって「C++ 化 = 速い」は必ずしも成立しません。C++ 化は自由度を得るための手段です。

## このプロジェクトの現状
- `pose_landmarker.cpp` はスタブ（検出なし）。環境変数でデバッグ出力可能
- 既存 ABI は Python 側 `py_native_pose.py` から利用済み
- 将来、`NPOSE_WITH_MEDIAPIPE` や `NPOSE_WITH_ONNXRUNTIME` を有効にして実装を差し替える設計

## 次のステップ（推奨）
1. 短期: 既存の Python Tasks で入力解像度/サンプリング最適化（FPS 向上を即時確認）
2. 並行: ONNX Runtime 版の試作（MoveNet など）で C++ DLL 経由の実効 FPS を把握
3. 長期: MediaPipe Tasks C++ 版に挑戦（Bazel ビルドの自動化と DLL 化）

---
補足: 具体的な Bazel/CMake 設定のひな形や、ONNX/TFLite 実装雛形は必要に応じてこのフォルダに追記していきます。