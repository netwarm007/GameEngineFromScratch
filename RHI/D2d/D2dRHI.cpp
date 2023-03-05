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
    assert(m_fDestroyResourceHandler);
    m_fDestroyResourceHandler();

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

Vector2f D2dRHI::GetCanvasSize() const {
    auto rtSize = m_pRenderTarget->GetSize();
    return {rtSize.width, rtSize.height};
}
