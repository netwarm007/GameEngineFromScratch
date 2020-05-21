#pragma once
#include "WindowsApplication.hpp"

namespace My {
class OpenGLApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    int Initialize() override;
    void Finalize() override;
    void Tick() override;

   protected:
    void CreateMainWindow() override;

   private:
    HDC m_hDC;
    HGLRC m_RenderContext;
    int m_nPixelFormat;
    PIXELFORMATDESCRIPTOR m_pfd;
};
}  // namespace My
