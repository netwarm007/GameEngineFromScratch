#pragma once
#include <xcb/xcb.h>
#include <X11/Xlib-xcb.h>

#include "BaseApplication.hpp"

namespace My {
class XcbApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;

    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override {
        return reinterpret_cast<void*>(m_XWindow);
    };

    void CreateMainWindow() override;

    void GetFramebufferSize(uint32_t& width, uint32_t& height) override;

   protected:
    void FinalizePlatformResources() override;
    Display *m_pDisplay = nullptr;
    int m_nScreen;

    xcb_connection_t* m_pConn = nullptr;
    xcb_screen_t* m_pScreen = nullptr;
    xcb_window_t m_XWindow = XCB_WINDOW_NONE;
};
}  // namespace My
