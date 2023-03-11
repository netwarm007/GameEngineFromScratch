#include "D2dRHI.hpp"

#include <objbase.h>

#include "IApplication.hpp"
#include "utility.hpp"

using namespace My;

HRESULT D2dRHI::CreateGraphicsResources() {
    HRESULT hr = S_OK;

    HWND hWnd = m_fGetWindowHandler();

    if (m_pRenderTarget == nullptr) {
        RECT rc;
        GetClientRect(hWnd, &rc);

        D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);

        hr = m_pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(),
            D2D1::HwndRenderTargetProperties(hWnd, size), &m_pRenderTarget);
    }

    // Create client resource
    assert(m_fCreateResourceHandler);
    m_fCreateResourceHandler();

    return hr;
}

HRESULT D2dRHI::RecreateGraphicsResources() {
    HRESULT hr = S_OK;

    if (m_pRenderTarget) {
        // Destroy client resource
        assert(m_fDestroyResourceHandler);
        m_fDestroyResourceHandler();

        SafeRelease(&m_pRenderTarget);
        hr = CreateGraphicsResources();
    }

    return hr;
}

void D2dRHI::BeginFrame() {
    // start build GPU draw command
    m_pRenderTarget->BeginDraw();
}

void D2dRHI::EndFrame() {
    m_pRenderTarget->EndDraw();
}

void D2dRHI::DestroyAll() {
    // Destroy client resource
    if(m_fDestroyResourceHandler) {
        m_fDestroyResourceHandler();
    }

    SafeRelease(&m_pRenderTarget);
    SafeRelease(&m_pFactory);
}

D2dRHI::D2dRHI() {
    // initialize COM
    if (FAILED(CoInitializeEx(
            nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) {
        assert(0);
    }

    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pFactory))) {
        assert(0);
    }
}

D2dRHI::~D2dRHI() {
    CoUninitialize();
}

ID2D1SolidColorBrush* D2dRHI::CreateSolidColorBrush(Vector3f color) const {
    ID2D1SolidColorBrush* result;
    m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(color[0], color[1], color[2]), &result);

    return result;
}

void D2dRHI::ClearCanvas(Vector4f color) const {
    // clear the background with clear color
    m_pRenderTarget->Clear(D2D1::ColorF(color[0], color[1], color[2], color[3]));
}

void D2dRHI::DrawLine(My::Point2Df start, My::Point2Df end, ID2D1SolidColorBrush* brush, float line_width) const {
    m_pRenderTarget->DrawLine(
        D2D1::Point2F(start[0], start[1]),
        D2D1::Point2F(end[0], end[1]),
        brush,
        line_width
        );
}

static DXGI_FORMAT getDxgiFormat(const Image& img) {
    DXGI_FORMAT format;

    if (img.compressed) {
        switch (img.compress_format) {
            case COMPRESSED_FORMAT::BC1:
            case COMPRESSED_FORMAT::DXT1:
                format = ::DXGI_FORMAT_BC1_UNORM;
                break;
            case COMPRESSED_FORMAT::BC2:
            case COMPRESSED_FORMAT::DXT3:
                format = ::DXGI_FORMAT_BC2_UNORM;
                break;
            case COMPRESSED_FORMAT::BC3:
            case COMPRESSED_FORMAT::DXT5:
                format = ::DXGI_FORMAT_BC3_UNORM;
                break;
            case COMPRESSED_FORMAT::BC4:
                format = ::DXGI_FORMAT_BC4_UNORM;
                break;
            case COMPRESSED_FORMAT::BC5:
                format = ::DXGI_FORMAT_BC5_UNORM;
                break;
            case COMPRESSED_FORMAT::BC6H:
                format = ::DXGI_FORMAT_BC6H_UF16;
                break;
            case COMPRESSED_FORMAT::BC7:
                format = ::DXGI_FORMAT_BC7_UNORM;
                break;
            default:
                assert(0);
        }
    } else {
        switch (img.pixel_format) {
            case PIXEL_FORMAT::R8:
                format = ::DXGI_FORMAT_R8_UNORM;
                break;
            case PIXEL_FORMAT::RG8:
                format = ::DXGI_FORMAT_R8G8_UNORM;
                break;
            case PIXEL_FORMAT::RGBA8:
                format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
                break;
            case PIXEL_FORMAT::RGBA16:
                format = ::DXGI_FORMAT_R16G16B16A16_FLOAT;
                break;
            default:
                assert(0);
        }
    }

    return format;
}

ID2D1Bitmap* My::D2dRHI::CreateBitmap(const My::Image& img) const {
    ID2D1Bitmap *bitmap;
    D2D1_BITMAP_PROPERTIES bitmap_prop;
    bitmap_prop.dpiX = 150;
    bitmap_prop.dpiY = 150;
    bitmap_prop.pixelFormat.alphaMode = D2D1_ALPHA_MODE_IGNORE;
    bitmap_prop.pixelFormat.format = getDxgiFormat(img);

    m_pRenderTarget->CreateBitmap(D2D1::SizeU(img.Width, img.Height), bitmap_prop, &bitmap);
    bitmap->CopyFromMemory(nullptr, img.data, img.pitch);

    return bitmap;
}

void My::D2dRHI::UpdateBitmap(const My::Image& img,
                                      ID2D1Bitmap* bmp) const {
    bmp->CopyFromMemory(nullptr, img.data, img.pitch);
}

void My::D2dRHI::DrawBitmap(My::Point2Df left_top, My::Point2Df right_bottom,
                           ID2D1Bitmap *bitmap) const {
    D2D1_RECT_F destinationRectangle = D2D1::RectF(left_top[0], left_top[1], right_bottom[0], right_bottom[1]);
    m_pRenderTarget->DrawBitmap(
        bitmap,
        destinationRectangle
    );
}

Vector2f D2dRHI::GetCanvasSize() const {
    auto rtSize = m_pRenderTarget->GetSize();
    return {rtSize.width, rtSize.height};
}
