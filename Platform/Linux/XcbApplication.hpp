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

    private:
        xcb_connection_t*    m_pConn;
        xcb_screen_t*        m_pScreen;
        xcb_window_t		 m_Window;
        xcb_generic_event_t* m_pEvent;
    };
}

