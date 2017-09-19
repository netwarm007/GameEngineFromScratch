#include <xcb/xcb.h>
#include "BaseApplication.hpp"

namespace My {
    class XcbApplication : public BaseApplication
    {
    public:
        XcbApplication(GfxConfiguration& config)
            : BaseApplication(config) {};

        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();

    protected:
        xcb_connection_t*    m_pConn   = nullptr;
        xcb_screen_t*        m_pScreen = nullptr;
        xcb_window_t         m_Window;
        uint32_t             m_nVi     = 0;;
    };
}

