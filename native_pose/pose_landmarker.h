#pragma once
#include <stdint.h>

#ifdef _WIN32
  #ifdef NATIVE_POSE_EXPORTS
    #define NPOSE_API __declspec(dllexport)
  #else
    #define NPOSE_API __declspec(dllimport)
  #endif
#else
  #define NPOSE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 戻り値: 0=OK, <0=エラー
// エラーコード
// -1: invalid argument
// -2: not initialized / model load failed
// -3: internal error

// ハンドル
typedef void* npose_handle_t;

// 正規化ランドマーク
typedef struct {
  float x; // [0,1]
  float y; // [0,1]
  float z; // 任意スケール（0でも可）
  float visibility; // [0,1]
} npose_lm_t;

// 検出結果
typedef struct {
  int has_landmarks;      // 0 or 1
  int landmark_count;     // 実際の個数
  npose_lm_t* landmarks;  // landmark_count 分の配列（ライブラリ所有）
} npose_result_t;

// 作成/破棄
NPOSE_API int npose_create(const char* model_path, int num_threads, npose_handle_t* out);
NPOSE_API void npose_destroy(npose_handle_t handle);

// 推論（BGRA, row-major, stride=width*4, width/heightは画素単位）
NPOSE_API int npose_detect(npose_handle_t handle,
                           const uint8_t* bgra,
                           int width,
                           int height,
                           int stride,
                           npose_result_t* out_result);

#ifdef __cplusplus
}
#endif
