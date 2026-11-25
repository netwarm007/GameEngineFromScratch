#pragma once
#include <xcb/xcb.h>
#include <X11/Xlib-xcb.h>

#include "BaseApplication.hpp"
#include "IPlatformHost.hpp"

namespace My {
class XcbApplication : public BaseApplication, _implements_ IPlatformHost {
   public:
    using BaseApplication::BaseApplication;

    // One cycle of the main loop
    void Tick() override;

    void ProcessEvents() override;

    void* GetMainWindowHandler() override {
        return reinterpret_cast<void*>(m_XWindow);
    };

    void CreateMainWindow() override;

    void GetFramebufferSize(uint32_t& width, uint32_t& height) override;

    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void FinalizePlatformResources() override;

    void SetActiveBackend(IRenderBackend* backend) override {
        m_pActiveBackend = backend;
    }

    xcb_connection_t* GetXcbConnection() { return m_pConn; }
    void SetWindowTitle(const char* title) override;

   protected:
    Display *m_pDisplay = nullptr;
    int m_nScreen;
    IRenderBackend* m_pActiveBackend = nullptr;
    bool m_HasPreservedGeometry = false;
    int16_t m_PreservedX = 20, m_PreservedY = 20;
    uint16_t m_PreservedW = 0, m_PreservedH = 0;

    xcb_connection_t* m_pConn = nullptr;
    xcb_screen_t* m_pScreen = nullptr;
    xcb_window_t m_XWindow = XCB_WINDOW_NONE;
};
}  // namespace My
