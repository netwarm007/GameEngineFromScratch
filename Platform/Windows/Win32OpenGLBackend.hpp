#pragma once

#include <windows.h>

#include "IRenderBackend.hpp"

namespace My {
class IApplication;
class IPlatformHost;

// Session-owned Win32 OpenGL (WGL) backend lifecycle against a persistent
// Win32 host (M4.1 detachment). The pixel format is window-lifetime bound,
// so reconstruction/switching recreates the native window.
class Win32OpenGLBackend : _implements_ IRenderBackend {
   public:
    Win32OpenGLBackend(IApplication& application, IPlatformHost& platform);

    void CreateMainWindow() override;
    int InitializeGraphicsResources() override;
    void WaitForGraphicsIdle() override;
    void FinalizeGraphicsResources() override;
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void PresentFrame() override;
    void BeginFrame() override;
    void OnSessionActivated() override;
    bool RequiresWindowRecreation() const override { return true; }

   private:
    IApplication& m_Application;
    IPlatformHost& m_Platform;
    HDC m_hDC = nullptr;
    HGLRC m_RenderContext = nullptr;
    int m_nPixelFormat = 0;
    PIXELFORMATDESCRIPTOR m_pfd{};
};
}  // namespace My