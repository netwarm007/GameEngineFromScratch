#include "D3d12Backend.hpp"

#include <cstdio>
#include <system_error>

#include "IApplication.hpp"
#include "IPlatformHost.hpp"

using namespace My;

D3d12Backend::D3d12Backend(IApplication& application, IPlatformHost& platform)
    : m_Application(application), m_Platform(platform) {
    m_CreateFenceEvent = [] {
        return CreateEvent(nullptr, FALSE, FALSE, nullptr);
    };
}

void D3d12Backend::CreateMainWindow() { m_Application.CreateMainWindow(); }

int D3d12Backend::InitializeGraphicsResources() {
    if (!m_Application.GetMainWindowHandler()) {
        fprintf(stderr, "D3D12 initialization requires a native window\n");
        return -1;
    }

    auto getFramebufferSize = [this](uint32_t& width, uint32_t& height) {
        m_Application.GetFramebufferSize(width, height);
    };

    auto getWindowHandler = [this]() -> HWND {
        return (HWND)m_Application.GetMainWindowHandler();
    };

    auto getGfxConfigHandler = [this]() { return m_Application.GetConfiguration(); };

    m_Rhi.SetFramebufferSizeQueryCB(getFramebufferSize);
    m_Rhi.SetGetWindowHandlerCB(getWindowHandler);
    m_Rhi.SetGetGfxConfigCB(getGfxConfigHandler);

    try {
        m_Rhi.CreateDevice();
        m_Rhi.EnableDebugLayer();
        m_Rhi.CreateCommandQueues();
        m_Rhi.CreateSwapChain();
        m_Rhi.CreateSyncObjects([this] { return m_CreateFenceEvent(); });
        m_Rhi.CreateRenderTargets();
        m_Rhi.CreateDepthStencils();
        m_Rhi.CreateFramebuffers();
        m_Rhi.CreateCommandPools();
        m_Rhi.CreateCommandLists();
    } catch (const std::system_error& error) {
        fprintf(stderr, "D3D12 initialization failed: %s\n", error.what());
        m_Rhi.DestroyAll();
        return error.code().value();
    }
    return 0;
}

void D3d12Backend::WaitForGraphicsIdle() { m_Rhi.WaitForIdle(); }

void D3d12Backend::FinalizeGraphicsResources() { m_Rhi.DestroyAll(); }

int D3d12Backend::InitializeGuiPlatform() {
    return m_Platform.InitializeGuiPlatform();
}

void D3d12Backend::FinalizeGuiPlatform() { m_Platform.FinalizeGuiPlatform(); }

void D3d12Backend::OnWindowResize(int new_width, int new_height) {
    if (new_width > 0 && new_height > 0) m_Rhi.RecreateSwapChain();
}