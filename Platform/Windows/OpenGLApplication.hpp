#pragma once
#include "WindowsApplication.hpp"

namespace My {
class OpenGLApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    void Tick() override;

    void CreateMainWindow() override;

   private:
    void FinalizeGraphicsResources() override;
    HDC m_hDC = nullptr;
    HGLRC m_RenderContext = nullptr;
    int m_nPixelFormat;
    PIXELFORMATDESCRIPTOR m_pfd;
};
}  // namespace My
