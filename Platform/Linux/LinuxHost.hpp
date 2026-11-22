#pragma once

#include <SDL2/SDL.h>
#include <xcb/xcb.h>
#include <X11/Xlib-xcb.h>

#include "BaseApplication.hpp"
#include "IPlatformHost.hpp"

namespace My {
// Persistent dual-mode Linux host (M4.1): one application object owns either
// an SDL window (OpenGL, full ImGui platform integration) or an XCB window
// (Vulkan, tracked GUI limits). The object persists across backend switches;
// switching modes recreates the native window through the usual teardown/
// creation hooks. XCB ImGui parity remains the tracked follow-on.
class LinuxHost : public BaseApplication, public IPlatformHost {
   public:
    enum class Mode { Sdl, Xcb };

    LinuxHost();
    explicit LinuxHost(GfxConfiguration& cfg);

    // Select the windowing mode before CreateMainWindow.
    void SetMode(Mode mode) { m_Mode = mode; }
    Mode GetMode() const { return m_Mode; }

    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void ProcessEvents() override;

    void* GetMainWindowHandler() override;
    void CreateMainWindow() override;
    void GetFramebufferSize(uint32_t& width, uint32_t& height) override;

    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void FinalizePlatformResources() override;
    void ShutdownPlatform() override;

    void SetActiveBackend(IRenderBackend* backend) override {
        m_pActiveBackend = backend;
    }

    // Native accessors for the paired backend's surface factory.
    xcb_connection_t* GetXcbConnection() { return m_pConn; }
    SDL_Window* GetSdlWindow() { return m_pWindow; }
    void SetWindowFlags(Uint32 flags) { m_WindowFlags = flags; }

   protected:
    void onResize(int width, int height);

   private:
    void CreateSdlWindow();
    void CreateXcbWindow();
    void ProcessSdlEvents();
    void ProcessXcbEvents();

    Mode m_Mode = Mode::Sdl;
    IRenderBackend* m_pActiveBackend = nullptr;

    // SDL mode state
    SDL_Window* m_pWindow = nullptr;
    Uint32 m_WindowFlags = SDL_WINDOW_SHOWN;
    bool m_bInDrag = false;
    bool m_bSdlInitialized = false;

    // XCB mode state
    Display* m_pDisplay = nullptr;
    int m_nScreen = 0;
    xcb_connection_t* m_pConn = nullptr;
    xcb_screen_t* m_pScreen = nullptr;
    xcb_window_t m_XWindow = XCB_WINDOW_NONE;
};
}  // namespace My