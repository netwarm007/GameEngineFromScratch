#include "D3d12Application.hpp"

using namespace My;

void D3d12Application::CreateMainWindow() {
    WindowsApplication::CreateMainWindow();

    auto getFramebufferSize = [this](int& width, int& height) {
        GetFramebufferSize(width, height);
    };

    auto getWindowHandler = [this]() -> HWND {
        return (HWND)GetMainWindowHandler();
    };

    auto getGfxConfigHandler = [this]() { return GetConfiguration(); };

    // 设置回调函数
    m_Rhi.SetFramebufferSizeQueryCB(getFramebufferSize);
    m_Rhi.SetGetWindowHandlerCB(getWindowHandler);
    m_Rhi.SetGetGfxConfigCB(getGfxConfigHandler);
}

void D3d12Application::onWindowResize(int, int) { m_Rhi.RecreateSwapChain(); }
