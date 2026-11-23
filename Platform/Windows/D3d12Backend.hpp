#pragma once

#include <functional>

#include "D3d/D3d12RHI.hpp"
#include "IRenderBackend.hpp"

namespace My {
class IApplication;
class IPlatformHost;

// Session-owned D3D12 backend lifecycle against a persistent Win32 host
// (M4.1 detachment). The host keeps window ownership and event routing;
// this object owns the RHI and its native state.
class D3d12Backend : _implements_ IRenderBackend {
   public:
    D3d12Backend(IApplication& application, IPlatformHost& platform);

    D3d12RHI& GetRHI() { return m_Rhi; }

    // Legacy hosts inject their fence-event factory (testable failure path).
    void SetFenceEventFactory(std::function<HANDLE()> factory) {
        m_CreateFenceEvent = std::move(factory);
    }

    void CreateMainWindow() override;
    int InitializeGraphicsResources() override;
    void WaitForGraphicsIdle() override;
    void FinalizeGraphicsResources() override;
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void OnWindowResize(int new_width, int new_height) override;

   private:
    IApplication& m_Application;
    IPlatformHost& m_Platform;
    D3d12RHI m_Rhi;
    std::function<HANDLE()> m_CreateFenceEvent;
};
}  // namespace My