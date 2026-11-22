#pragma once
#include "IRenderBackend.hpp"
#include "SDLOpenGLBackend.hpp"
#include "SdlApplication.hpp"

namespace My {
// Legacy compile-time application: the SDL host plus an owned OpenGL
// backend. The runtime path composes the host and backend separately.
class OpenGLApplication : public SdlApplication, public IRenderBackend {
   public:
    OpenGLApplication();
    explicit OpenGLApplication(GfxConfiguration& cfg);

    void Tick() override {
        SdlApplication::Tick();
        PresentFrame();
    }

    void CreateMainWindow() override { m_Backend.CreateMainWindow(); }

   private:
    int InitializeGraphicsResources() override {
        return m_Backend.InitializeGraphicsResources();
    }
    int InitializeGuiPlatform() override {
        return m_Backend.InitializeGuiPlatform();
    }
    void FinalizeGuiPlatform() override { m_Backend.FinalizeGuiPlatform(); }
    void WaitForGraphicsIdle() override {}
    void FinalizeGraphicsResources() override {
        m_Backend.FinalizeGraphicsResources();
    }
    void PresentFrame() override { m_Backend.PresentFrame(); }
    bool RequiresWindowRecreation() const override { return true; }

    SDLOpenGLBackend m_Backend;
};
}  // namespace My
