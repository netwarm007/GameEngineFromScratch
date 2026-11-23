#pragma once
#include "D3d12Backend.hpp"
#include "WindowsApplication.hpp"

#include "config.h"

namespace My {
// Legacy compile-time application: the Win32 host plus an owned D3D12
// backend. The runtime path composes the host and backend separately; the
// backend logic lives in D3d12Backend for both paths.
class D3d12Application : public WindowsApplication, _implements_ IRenderBackend {
   public:
    D3d12Application() : m_Backend(*this, *this) { BindFenceFactory(); }
    explicit D3d12Application(GfxConfiguration& cfg)
        : WindowsApplication(cfg), m_Backend(*this, *this) {
        BindFenceFactory();
    }

    D3d12RHI& GetRHI() { return m_Backend.GetRHI(); }

    void CreateMainWindow() override { WindowsApplication::CreateMainWindow(); }
    int InitializeGuiPlatform() override {
        return WindowsApplication::InitializeGuiPlatform();
    }
    void FinalizeGuiPlatform() override {
        WindowsApplication::FinalizeGuiPlatform();
    }

   protected:
    virtual HANDLE CreateFenceEvent() {
        return CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }

   private:
    void BindFenceFactory() {
        m_Backend.SetFenceEventFactory([this] { return CreateFenceEvent(); });
    }

    int InitializeGraphicsResources() final {
        return m_Backend.InitializeGraphicsResources();
    }
    void WaitForGraphicsIdle() final { m_Backend.WaitForGraphicsIdle(); }
    void FinalizeGraphicsResources() final {
        m_Backend.FinalizeGraphicsResources();
    }
    void onWindowResize(int new_width, int new_height) final {
        m_Backend.OnWindowResize(new_width, new_height);
    }

    D3d12Backend m_Backend;
};
}  // namespace My
