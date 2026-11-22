#pragma once
#include "IRenderBackend.hpp"
#include "SdlApplication.hpp"

namespace My {
class OpenGLApplication : public SdlApplication, public IRenderBackend {
   public:
    using SdlApplication::SdlApplication;

    void Tick() override;

    void CreateMainWindow() override;

   private:
    // The SDL host couples the GL context to the window lifetime: the
    // context is created with the window and released here before the
    // window is destroyed.
    int InitializeGraphicsResources() override;
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void WaitForGraphicsIdle() override {}
    void FinalizeGraphicsResources() override;
    void PresentFrame() override;

    SDL_GLContext m_hContext = nullptr;
};
}  // namespace My
