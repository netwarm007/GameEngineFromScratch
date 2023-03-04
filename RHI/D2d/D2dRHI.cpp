#include "D2dRHI.hpp"

#include <objbase.h>

#include "IApplication.hpp"
#include "utility.hpp"

using namespace My;

HRESULT D2dRHI::CreateGraphicsResources() {
    HRESULT hr = S_OK;

    // initialize COM
    if (FAILED(hr = CoInitializeEx(
            nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)))
        return hr;

    if (FAILED(
            hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pFactory)))
        return hr;

    HWND hWnd = m_fGetWindowHandler();

    if (m_pRenderTarget == nullptr) {
        RECT rc;
        GetClientRect(hWnd, &rc);

        D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);

        hr = m_pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(),
            D2D1::HwndRenderTargetProperties(hWnd, size), &m_pRenderTarget);
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
    SafeRelease(&m_pRenderTarget);
    SafeRelease(&m_pFactory);

    CoUninitialize();
}

D2dRHI::D2dRHI() {
}

D2dRHI::~D2dRHI() {
}

ID2D1SolidColorBrush* D2dRHI::CreateSolidColorBrush(Vector3f color) {
    ID2D1SolidColorBrush* result;
    m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(color[0], color[1], color[2]), &result);

    return result;
}

void D2dRHI::ClearCanvas(Vector4f color) {
    // clear the background with clear color
    m_pRenderTarget->Clear(D2D1::ColorF(color[0], color[1], color[2], color[3]));
}