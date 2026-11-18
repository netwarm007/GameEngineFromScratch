#pragma once
#include "WindowsApplication.hpp"

namespace My {
class OpenGLApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    int Initialize() override;
    void Tick() override;

   private:
    int InitializeGuiPlatform() override;
    int InitializeGraphicsResources() override;
    void FinalizeGraphicsResources() override;
    HDC m_hDC = nullptr;
    HGLRC m_RenderContext = nullptr;
    int m_nPixelFormat;
    PIXELFORMATDESCRIPTOR m_pfd;
};
}  // namespace My
