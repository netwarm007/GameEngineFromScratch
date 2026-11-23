#pragma once

#include <SDL2/SDL.h>

#include <functional>

#include "IRenderBackend.hpp"

namespace My {
class IApplication;
class IPlatformHost;

// Session-owned SDL OpenGL backend lifecycle (M4.1 detachment). The GL
// context is window-lifetime bound, so reconstruction/switching recreates
// the native window. Window creation flags and the window accessor are
// injected so the backend pairs with either the legacy SDL application or
// the dual-mode Linux host.
class SDLOpenGLBackend : _implements_ IRenderBackend {
   public:
    SDLOpenGLBackend(IApplication& application, IPlatformHost& platform,
                     std::function<SDL_Window*()> windowProvider,
                     std::function<void(Uint32)> windowFlagsSetter);

    SDL_GLContext GetContext() { return m_hContext; }

    void CreateMainWindow() override;
    int InitializeGraphicsResources() override;
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void WaitForGraphicsIdle() override {}
    void FinalizeGraphicsResources() override;
    void PresentFrame() override;
    void BeginFrame() override;
    bool RequiresWindowRecreation() const override { return true; }

   private:
    IApplication& m_Application;
    IPlatformHost& m_Platform;
    std::function<SDL_Window*()> m_WindowProvider;
    std::function<void(Uint32)> m_WindowFlagsSetter;
    SDL_GLContext m_hContext = nullptr;
};
}  // namespace My