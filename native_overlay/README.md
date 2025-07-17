# native_overlay (DirectWrite/Direct2D overlay for OpenCV frames)

This module provides a native Windows renderer (C++/DirectWrite + Direct2D) to draw Japanese text into OpenCV frames efficiently. It exposes a single C API function to draw multiple text items into a given BGRA buffer, intended to be called from Python via `ctypes`.

- Build: MSVC (Desktop), x64. Requires Windows SDK.
- Runtime: Windows 10+.

## API (C)

```
struct TextItem {
    int x;
    int y;
    int fontSize;
    unsigned int colorARGB; // 0xAARRGGBB
    const wchar_t* text;    // wide string (UTF-16)
};

// Draws N text items into the image buffer.
// imageData: BGRA buffer (height * stride)
// stride: bytes per row (typically width * 4)
// width, height: image size in pixels
// items: pointer to array of TextItem
// count: number of items
__declspec(dllexport) int __cdecl DrawTextOverlay(
    unsigned char* imageData,
    int stride,
    int width,
    int height,
    const TextItem* items,
    int count
);
```

## Notes

- The buffer must be BGRA (premultiplied alpha not required; we blend opaque text).
- The renderer initializes D2D/DWrite factories on first call and reuses them.
- Thread-safe by single-thread use; for multi-threading, guard externally.

## Python usage (ctypes)

- See `py_native_overlay.py` for a minimal Python wrapper.

## Build instructions (Windows, Visual Studio)

1) Install Visual Studio 2022 Build Tools with "Desktop development with C++" and Windows 10 SDK.
2) In VS Code, run the task "Build overlay.dll (MSBuild)" or execute:
     - PowerShell:
         ```powershell
         powershell -NoProfile -ExecutionPolicy Bypass -File native_overlay/build_msbuild.ps1 -Configuration Release
         ```
3) The DLL will be at `native_overlay/x64/Release/overlay.dll`.

## Build instructions (CMake/Ninja alternative)

If you have CMake and Ninja installed:

```powershell
pwsh native_overlay/build.ps1 -BuildType Release -Generator Ninja
```

Output DLL will be under `native_overlay/build/`.
