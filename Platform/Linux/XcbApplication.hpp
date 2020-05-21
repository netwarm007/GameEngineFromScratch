#include <xcb/xcb.h>

#include "BaseApplication.hpp"

namespace My {
class XcbApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;

    void Finalize() override;
    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override {
        return reinterpret_cast<void*>(m_Window);
    };

   protected:
    void CreateMainWindow() override;

   protected:
    xcb_connection_t* m_pConn = nullptr;
    xcb_screen_t* m_pScreen = nullptr;
    xcb_window_t m_Window;
    uint32_t m_nVi = 0;
};
}  // namespace My
