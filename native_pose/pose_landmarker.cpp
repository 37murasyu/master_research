#include "pose_landmarker.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// 簡易ヘルパ: 環境変数の真偽/数値を読む
static bool env_truthy(const char* name) {
  const char* v = std::getenv(name);
  if (!v) return false;
  // "0", "", "false", "off" 以外は真と扱う
  if (v[0] == '\0') return false;
  if (v[0] == '0' && v[1] == '\0') return false;
  std::string s(v);
  for (auto& c : s) c = static_cast<char>(::tolower(c));
  return !(s == "false" || s == "off" || s == "no");
}

static int env_int(const char* name, int defVal) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') return defVal;
  try {
    return std::stoi(v);
  } catch (...) {
    return defVal;
  }
}

struct PoseImpl {
  std::string model;
  int threads{0};
  std::vector<npose_lm_t> last;
  // debug
  bool trace{false};
  int trace_every{10};
  unsigned long long call_count{0};
  const char* backend_name{
#if defined(NPOSE_WITH_MEDIAPIPE)
    "mediapipe"
#elif defined(NPOSE_WITH_ONNXRUNTIME)
    "onnxruntime"
#else
    "stub"
#endif
  };
};

extern "C" {

NPOSE_API int npose_create(const char* model_path, int num_threads, npose_handle_t* out) {
  if (!out) return -1;
  try {
    auto impl = new PoseImpl();
    if (model_path) impl->model = model_path;
    impl->threads = num_threads;
    // debug 設定
    impl->trace = env_truthy("NPOSE_TRACE");
    impl->trace_every = env_int("NPOSE_TRACE_EVERY", 10);
    if (impl->trace) {
      bool exists = false;
      if (model_path) {
        std::ifstream f(model_path, std::ios::binary);
        exists = f.good();
      }
      std::fprintf(stderr,
                   "[NPOSE] create | backend=%s model='%s' exists=%d threads=%d\n",
                   impl->backend_name,
                   model_path ? model_path : "(null)",
                   exists ? 1 : 0,
                   num_threads);
      std::fprintf(stderr,
                   "[NPOSE] NOTE: current backend is '%s'%s\n",
                   impl->backend_name,
                   std::string(impl->backend_name) == "stub" ? " (returns no landmarks)" : "");
    }
    *out = reinterpret_cast<npose_handle_t>(impl);
    return 0; // いまはスタブ（将来ここでMediaPipe初期化）
  } catch (...) {
    return -3;
  }
}

NPOSE_API void npose_destroy(npose_handle_t handle) {
  try {
    auto impl = reinterpret_cast<PoseImpl*>(handle);
    if (impl && impl->trace) {
      std::fprintf(stderr, "[NPOSE] destroy | model='%s' calls=%llu\n",
                   impl->model.c_str(),
                   impl->call_count);
    }
    delete impl;
  } catch (...) {
    // no-throw
  }
}

NPOSE_API int npose_detect(npose_handle_t handle,
                 const uint8_t* bgra,
                 int width,
                 int height,
                 int stride,
                 npose_result_t* out_result) {
  if (!handle || !out_result) return -1;
  try {
    auto impl = reinterpret_cast<PoseImpl*>(handle);
    impl->call_count++;
#if defined(NPOSE_WITH_MEDIAPIPE)
    // TODO: MediaPipe Tasks C++ 実装
    // - ここで BGRA → RGB 変換（必要なら）
    // - Image/Frame に詰めてポーズ推定器を呼ぶ
    // - npose_result_t に 33 点分を正規化座標で格納
    // 現状は未実装のためフォールバックで stub を返す
#elif defined(NPOSE_WITH_ONNXRUNTIME)
    // TODO: ONNX Runtime 実装
    // - BGRA → NCHW/NHWC テンソル整形
    // - 推論 → キーポイント抽出
    // - MediaPipe 互換の正規化/可視度に合わせて out_result を定義
    // 現状は未実装のためフォールバックで stub を返す
#endif
    // 画像の基本情報を出す（トレース有効時のみ、間引きあり）
    if (impl->trace) {
      // 引数名を維持しつつログだけに使うため、再度シグネチャを宣言せずコメントのままにする
      // ここではスタブで検出を返さないことを明示
      if (impl->call_count % static_cast<unsigned long long>(impl->trace_every) == 1ull) {
        std::fprintf(stderr,
                     "[NPOSE] detect | call=%llu backend=%s (stub) w=%d h=%d stride=%d buf=%p\n",
                     impl->call_count, impl->backend_name, width, height, stride, (const void*)bgra);
      }
    }
    // スタブ: 検出なし
    impl->last.clear();
    out_result->has_landmarks = 0;
    out_result->landmark_count = 0;
    out_result->landmarks = nullptr;
    return 0;
  } catch (...) {
    return -3;
  }
}

} // extern "C"
