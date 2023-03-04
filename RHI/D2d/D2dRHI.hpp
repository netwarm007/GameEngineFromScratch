#pragma once
#include <d2d1.h>
#include <functional>

#include "GfxConfiguration.hpp"
#include "geommath.hpp"

namespace My {
class D2dRHI {
   public:
    using QueryFrameBufferSizeFunc = std::function<void(uint32_t&, uint32_t&)>;
    using GetWindowHandlerFunc = std::function<HWND()>;
    using GetGfxConfigFunc = std::function<const GfxConfiguration&()>;
    using DestroyResourceFunc = std::function<void()>;

   public:
    D2dRHI();
    ~D2dRHI();

   public:
    void SetFramebufferSizeQueryCB(const QueryFrameBufferSizeFunc& func) {
        m_fQueryFramebufferSize = func;
    }
    void SetGetWindowHandlerCB(const GetWindowHandlerFunc& func) {
        m_fGetWindowHandler = func;
    }
    void SetGetGfxConfigCB(const GetGfxConfigFunc& func) {
        m_fGetGfxConfigHandler = func;
    }
    void DestroyResourceCB(const DestroyResourceFunc& func) {
        m_fDestroyResourceHandler = func;
    }

    HRESULT CreateGraphicsResources();

    void BeginFrame();
    void EndFrame();

    ID2D1SolidColorBrush* CreateSolidColorBrush(Vector3f color);

    void ClearCanvas(Vector4f color);

    void DestroyAll();

   protected:
    ID2D1Factory *m_pFactory = nullptr;

    ID2D1HwndRenderTarget *m_pRenderTarget = nullptr;

   private:
    QueryFrameBufferSizeFunc m_fQueryFramebufferSize;
    GetWindowHandlerFunc m_fGetWindowHandler;
    GetGfxConfigFunc m_fGetGfxConfigHandler;
    DestroyResourceFunc m_fDestroyResourceHandler;
};
}  // namespace My
