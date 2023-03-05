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
    using CreateResourceFunc = std::function<void()>;
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
    void CreateResourceCB(const CreateResourceFunc& func) {
        m_fCreateResourceHandler = func;
    }
    void DestroyResourceCB(const DestroyResourceFunc& func) {
        m_fDestroyResourceHandler = func;
    }

    HRESULT CreateGraphicsResources();
    HRESULT RecreateGraphicsResources();

    void BeginFrame();
    void EndFrame();

    ID2D1SolidColorBrush* CreateSolidColorBrush(Vector3f color) const;
    void ClearCanvas(Vector4f color) const;
    Vector2f GetCanvasSize() const;
    void DrawLine(My::Point2Df start, My::Point2Df end, ID2D1SolidColorBrush* brush, float line_width) const;

    void DestroyAll();

   protected:
    ID2D1Factory *m_pFactory = nullptr;

    ID2D1HwndRenderTarget *m_pRenderTarget = nullptr;

   private:
    QueryFrameBufferSizeFunc m_fQueryFramebufferSize;
    GetWindowHandlerFunc m_fGetWindowHandler;
    GetGfxConfigFunc m_fGetGfxConfigHandler;
    CreateResourceFunc m_fCreateResourceHandler;
    DestroyResourceFunc m_fDestroyResourceHandler;
};
}  // namespace My
