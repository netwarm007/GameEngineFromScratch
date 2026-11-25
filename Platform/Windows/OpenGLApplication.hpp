#pragma once
#include "IRenderBackend.hpp"
#include "Win32OpenGLBackend.hpp"
#include "WindowsApplication.hpp"

namespace My {
// Legacy compile-time application: the Win32 host plus an owned WGL OpenGL
// backend. The runtime path composes the host and backend separately.
class OpenGLApplication : public WindowsApplication, _implements_ IRenderBackend {
   public:
    OpenGLApplication() : m_Backend(*this, *this) {}
    explicit OpenGLApplication(GfxConfiguration& cfg)
        : WindowsApplication(cfg), m_Backend(*this, *this) {}

    int Initialize() override {
        const int result = WindowsApplication::Initialize();
        if (result == 0) m_Backend.OnSessionActivated();
        return result;
    }
    void Tick() override {
        WindowsApplication::Tick();
        PresentFrame();
    }

    void CreateMainWindow() override { WindowsApplication::CreateMainWindow(); }

   private:
    int InitializeGuiPlatform() override {
        return m_Backend.InitializeGuiPlatform();
    }
    void FinalizeGuiPlatform() override { WindowsApplication::FinalizeGuiPlatform(); }
    int InitializeGraphicsResources() override {
        return m_Backend.InitializeGraphicsResources();
    }
    void WaitForGraphicsIdle() override { m_Backend.WaitForGraphicsIdle(); }
    void FinalizeGraphicsResources() override {
        m_Backend.FinalizeGraphicsResources();
    }
    void PresentFrame() override { m_Backend.PresentFrame(); }
    bool RequiresWindowRecreation() const override { return true; }

    Win32OpenGLBackend m_Backend;
};
}  // namespace My
