#pragma once
#include <stdint.h>

#ifdef _WIN32
  #ifdef NATIVE_OVERLAY_EXPORTS
    #define NATIVE_API __declspec(dllexport)
  #else
    #define NATIVE_API __declspec(dllimport)
  #endif
#else
  #define NATIVE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TextItem {
    int32_t x;
    int32_t y;
    int32_t fontSize;
    uint32_t colorARGB; // 0xAARRGGBB
    const wchar_t* text; // UTF-16
} TextItem;

// Returns 0 on success, negative on failure
NATIVE_API int DrawTextOverlay(
    uint8_t* imageData,
    int32_t stride,
    int32_t width,
    int32_t height,
    const TextItem* items,
    int32_t count
);

#ifdef __cplusplus
}
#endif
