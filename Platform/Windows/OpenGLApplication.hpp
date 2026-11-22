#pragma once
#include "IRenderBackend.hpp"
#include "WindowsApplication.hpp"

namespace My {
class OpenGLApplication : public WindowsApplication, public IRenderBackend {
   public:
    using WindowsApplication::WindowsApplication;

    int Initialize() override;
    void Tick() override;

   private:
    int InitializeGuiPlatform() override;
    int InitializeGraphicsResources() override;
    void WaitForGraphicsIdle() override;
    void FinalizeGraphicsResources() override;
    void PresentFrame() override;
    HDC m_hDC = nullptr;
    HGLRC m_RenderContext = nullptr;
    int m_nPixelFormat;
    PIXELFORMATDESCRIPTOR m_pfd;
};
}  // namespace My
