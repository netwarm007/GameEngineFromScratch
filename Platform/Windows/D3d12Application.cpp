#include "D3d12Application.hpp"

using namespace My;

void D3d12Application::CreateMainWindow() {
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
    m_Rhi.CreateDevice();

    // 打开调试层（只对Debug版本有效）
    m_Rhi.EnableDebugLayer();

    // 创建命令提交队列
    m_Rhi.CreateCommandQueues();

    // 创建交换链
    m_Rhi.CreateSwapChain();

    // 创建同步对象
    m_Rhi.CreateSyncObjects();

    // 创建渲染目标缓冲区
    m_Rhi.CreateRenderTargets();

    // 创建深度和蒙板缓冲区
    m_Rhi.CreateDepthStencils();

    // 创建Framebuffer描述表
    m_Rhi.CreateFramebuffers();

    // 创建命令清单池
    m_Rhi.CreateCommandPools();

    // 创建命令列表
    m_Rhi.CreateCommandLists();
}

void D3d12Application::onWindowResize(int, int) { m_Rhi.RecreateSwapChain(); }

void D3d12Application::Finalize() {
    // 销毁相关资源
    m_Rhi.DestroyAll();

    WindowsApplication::Finalize();
}