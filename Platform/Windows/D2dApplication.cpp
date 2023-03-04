#include "D2dApplication.hpp"

using namespace My;

void D2dApplication::CreateMainWindow() {
    WindowsApplication::CreateMainWindow();

    auto getFramebufferSize = [this](uint32_t& width, uint32_t& height) {
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

    // 创建设备
    m_Rhi.CreateGraphicsResources();
}

void D2dApplication::onWindowResize(int, int) { ; }

void D2dApplication::Finalize() {
    // 销毁相关资源
    m_Rhi.DestroyAll();

    WindowsApplication::Finalize();
}