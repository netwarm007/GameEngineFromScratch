#include "D2dApplication.hpp"

#include <cstdio>

using namespace My;

int D2dApplication::InitializeGraphicsResources() {
    if (!GetMainWindowHandler()) {
        fprintf(stderr, "D2D initialization requires a native window\n");
        return -1;
    }

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
    return 0;
}

void D2dApplication::onWindowResize(int, int) { 
    m_Rhi.RecreateGraphicsResources();
}

void D2dApplication::FinalizeGraphicsResources() {
    // 销毁相关资源
    m_Rhi.DestroyAll();
}