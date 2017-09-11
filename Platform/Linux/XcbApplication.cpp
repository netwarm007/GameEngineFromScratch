#include <string.h>
#include "XcbApplication.hpp"
#include "MemoryManager.hpp"
#include "GraphicsManager.hpp"

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 32, 0, 0, 960, 540, "Game Engine From Scratch (XCB)");
    IApplication* g_pApp                  = static_cast<IApplication*>(new XcbApplication(config));
    GraphicsManager* g_pGraphicsManager   = static_cast<GraphicsManager*>(new GraphicsManager);
    MemoryManager*   g_pMemoryManager     = static_cast<MemoryManager*>(new MemoryManager);
}

int My::XcbApplication::Initialize()
{
    int result;
  	uint32_t		mask = 0;
	uint32_t		values[2];

    // first call base class initialization
    result = BaseApplication::Initialize();

    if (result != 0)
        exit(result);

	/* establish connection to X server */
	m_pConn = xcb_connect(0, 0);

	/* get the first screen */
	m_pScreen = xcb_setup_roots_iterator(xcb_get_setup(m_pConn)).data;

	/* get the root window */
	m_Window = m_pScreen->root;

	/* create window */
	m_Window = xcb_generate_id(m_pConn);
	mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	values[0] = m_pScreen->white_pixel;
	values[1] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;
	xcb_create_window (m_pConn,					/* connection */
					   XCB_COPY_FROM_PARENT,	/* depth */
					   m_Window,					/* window ID */
					   m_pScreen->root,			/* parent window */
					   20, 20,					/* x, y */
					   m_Config.screenWidth,    /* width */
                       m_Config.screenHeight,	/* height */
					   10,						/* boarder width */
					   XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class */
					   m_pScreen->root_visual,	/* visual */
					   mask, values);			/* masks */

	/* set the title of the window */
	xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_Window,
			    XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
			    strlen(m_Config.appName), m_Config.appName);

	/* set the title of the window icon */
    /*
	xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_Window,
			    XCB_ATOM_WM_ICON_NAME, XCB_ATOM_STRING, 8,
			    strlen(m_Config.appName), m_Config.appName);
    */

	/* map the window on the screen */
	xcb_map_window(m_pConn, m_Window);

	xcb_flush(m_pConn);

    return result;
}

void My::XcbApplication::Finalize()
{
    xcb_disconnect(m_pConn);
}

void My::XcbApplication::Tick()
{
    xcb_generic_event_t* pEvent;
    pEvent = xcb_wait_for_event(m_pConn);
    switch(pEvent->response_type & ~0x80) {
	case XCB_EXPOSE:
		    {		
		    }
			break;
	case XCB_KEY_PRESS:
            BaseApplication::m_bQuit = true;
			break;
	}
	free(pEvent);
}


