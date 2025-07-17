#include "overlay.h"

#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <wincodec.h> // WIC for offscreen render target
#include <vector>
#include <cwchar>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

static ID2D1Factory*        g_d2dFactory = nullptr;
static IDWriteFactory*      g_dwFactory = nullptr;
static IWICImagingFactory*  g_wicFactory = nullptr;

template <class T>
static void SafeRelease(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

static HRESULT EnsureFactories() {
    if (!g_d2dFactory) {
        D2D1_FACTORY_OPTIONS opts = {};
        // Use MULTI_THREADED for broader compatibility when called from Python threads
        HRESULT hr = D2D1CreateFactory(
            D2D1_FACTORY_TYPE_MULTI_THREADED,
            __uuidof(ID2D1Factory),
            &opts,
            reinterpret_cast<void**>(&g_d2dFactory)
        );
        if (FAILED(hr)) return hr;
    }
    if (!g_dwFactory) {
        HRESULT hr = DWriteCreateFactory(
            DWRITE_FACTORY_TYPE_SHARED,
            __uuidof(IDWriteFactory),
            reinterpret_cast<IUnknown**>(&g_dwFactory)
        );
        if (FAILED(hr)) return hr;
    }
    if (!g_wicFactory) {
        HRESULT hr = CoCreateInstance(
            CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
            __uuidof(IWICImagingFactory),
            reinterpret_cast<void**>(&g_wicFactory)
        );
        if (FAILED(hr)) return hr;
    }
    return S_OK;
}

extern "C" __declspec(dllexport) int DrawTextOverlay(
    uint8_t* imageData,
    int32_t stride,
    int32_t width,
    int32_t height,
    const TextItem* items,
    int32_t count)
{
    if (!imageData || stride <= 0 || width <= 0 || height <= 0 || (count > 0 && !items)) return -1;
    // Initialize COM for the calling thread if not already.
    // It's safe to call CoInitializeEx multiple times; the paired CoUninitialize is managed by the caller's process lifetime.
    (void)CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    HRESULT hr = EnsureFactories();
    if (FAILED(hr)) return -2;
    // 1) Create a WIC bitmap initialized from our input buffer (copy-in)
    IWICBitmap* wicBitmap = nullptr;
    hr = g_wicFactory->CreateBitmapFromMemory(
        (UINT)width,
        (UINT)height,
        // Direct2D WIC render targets generally expect premultiplied BGRA
        GUID_WICPixelFormat32bppPBGRA,
        (UINT)stride,
        (UINT)(stride * height),
        imageData,
        &wicBitmap
    );
    if (FAILED(hr)) return -3;

    // 2) Create a D2D render target that draws into that WIC bitmap
    D2D1_RENDER_TARGET_PROPERTIES rtp = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        // Match premultiplied BGRA pixel format
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
        0.0f, 0.0f,
        D2D1_RENDER_TARGET_USAGE_NONE,
        D2D1_FEATURE_LEVEL_DEFAULT
    );
    ID2D1RenderTarget* rt = nullptr;
    hr = g_d2dFactory->CreateWicBitmapRenderTarget(wicBitmap, &rtp, &rt);
    if (FAILED(hr)) return -4;

    rt->BeginDraw();

    // Prepare a solid color brush (updated per item)
    ID2D1SolidColorBrush* brush = nullptr;
    hr = rt->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 1), &brush);
    if (FAILED(hr)) { rt->EndDraw(); SafeRelease(rt); SafeRelease(wicBitmap); return -5; }

    for (int i = 0; i < count; ++i) {
        const TextItem& it = items[i];
        float a = ((it.colorARGB >> 24) & 0xFF) / 255.0f;
        float r = ((it.colorARGB >> 16) & 0xFF) / 255.0f;
        float g = ((it.colorARGB >> 8)  & 0xFF) / 255.0f;
        float b = ((it.colorARGB)       & 0xFF) / 255.0f;
        brush->SetColor(D2D1::ColorF(r, g, b, a));

        IDWriteTextFormat* fmt = nullptr;
        hr = g_dwFactory->CreateTextFormat(
            L"Meiryo",
            nullptr,
            DWRITE_FONT_WEIGHT_NORMAL,
            DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL,
            (FLOAT)it.fontSize,
            L"ja-jp",
            &fmt
        );
        if (FAILED(hr)) continue;
        fmt->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
        fmt->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);

        if (!it.text) continue;
        D2D1_RECT_F layout = D2D1::RectF(
            (FLOAT)it.x,
            (FLOAT)it.y,
            (FLOAT)width,
            (FLOAT)(it.y + it.fontSize * 2)
        );
    rt->DrawText(it.text, (UINT32)wcslen(it.text), fmt, &layout, brush);
        SafeRelease(fmt);
    }

    hr = rt->EndDraw();
    if (FAILED(hr)) { SafeRelease(brush); SafeRelease(rt); SafeRelease(wicBitmap); return -6; }

    // 3) Copy pixels back into caller-provided buffer
    hr = wicBitmap->CopyPixels(
        nullptr,
        (UINT)stride,
        (UINT)(stride * height),
        imageData
    );
    if (FAILED(hr)) { SafeRelease(brush); SafeRelease(rt); SafeRelease(wicBitmap); return -7; }

    SafeRelease(brush);
    SafeRelease(rt);
    SafeRelease(wicBitmap);

    return 0;
}
