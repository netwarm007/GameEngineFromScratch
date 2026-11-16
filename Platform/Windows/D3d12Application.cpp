#include "D3d12Application.hpp"

#include <cstdio>
#include <system_error>

using namespace My;

int D3d12Application::InitializeGraphicsResources() {
    if (!GetMainWindowHandler()) {
        fprintf(stderr, "D3D12 initialization requires a native window\n");
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

    try {
        // 创建设备
        m_Rhi.CreateDevice();

        // 打开调试层（只对Debug版本有效）
        m_Rhi.EnableDebugLayer();

        // 创建命令提交队列
        m_Rhi.CreateCommandQueues();

        // 创建交换链
        m_Rhi.CreateSwapChain();

        // 创建同步对象
        m_Rhi.CreateSyncObjects([this] { return CreateFenceEvent(); });

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
    } catch (const std::system_error& error) {
        fprintf(stderr, "D3D12 initialization failed: %s\n", error.what());
        m_Rhi.DestroyAll();
        return error.code().value();
    }
    return 0;
}

void D3d12Application::onWindowResize(int width, int height) {
    if (width > 0 && height > 0) m_Rhi.RecreateSwapChain();
}

void D3d12Application::WaitForGraphicsIdle() {
    m_Rhi.WaitForIdle();
}

void D3d12Application::FinalizeGraphicsResources() {
    m_Rhi.DestroyAll();
}