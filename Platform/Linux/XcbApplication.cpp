#include "XcbApplication.hpp"

#include <string.h>

#include <cstdio>
#include <iostream>

#include "InputManager.hpp"

using namespace My;
using namespace std;

void XcbApplication::CreateMainWindow() {
    uint32_t mask = 0;
    uint32_t values[3];

    if (!m_pConn) {
        /* establish connection to X server */
        m_pConn = xcb_connect(0, 0);
    }

    if (!m_pScreen) {
        /* get the first screen */
        m_pScreen = xcb_setup_roots_iterator(xcb_get_setup(m_pConn)).data;
        m_nVi = m_pScreen->root_visual;
    }

    /* get the root window */
    m_Window = m_pScreen->root;

    /* Create XID's for colormap */
    xcb_colormap_t colormap = xcb_generate_id(m_pConn);

    xcb_create_colormap(m_pConn, XCB_COLORMAP_ALLOC_NONE, colormap, m_Window,
                        m_nVi);

    /* create window */
    m_Window = xcb_generate_id(m_pConn);
    mask = XCB_CW_EVENT_MASK | XCB_CW_COLORMAP;
    values[0] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;
    values[1] = colormap;
    values[2] = 0;
    xcb_create_window(m_pConn,                       /* connection */
                      XCB_COPY_FROM_PARENT,          /* depth */
                      m_Window,                      /* window ID */
                      m_pScreen->root,               /* parent window */
                      20, 20,                        /* x, y */
                      m_Config.screenWidth,          /* width */
                      m_Config.screenHeight,         /* height */
                      10,                            /* boarder width */
                      XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class */
                      m_nVi,                         /* visual */
                      mask, values);                 /* masks */

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
}

void XcbApplication::Finalize() { xcb_disconnect(m_pConn); }

void XcbApplication::Tick() {
    xcb_generic_event_t* pEvent;
    if ((pEvent = xcb_poll_for_event(m_pConn))) {
        switch (pEvent->response_type & ~0x80) {
            case XCB_EXPOSE:
                break;
            case XCB_KEY_PRESS: {
                auto key_code =
                    reinterpret_cast<xcb_key_press_event_t*>(pEvent)->detail;
                printf("[XcbApplication] Key Press: Keycode: %d\n", key_code);
                switch (key_code) {
                    case 113:
                        g_pInputManager->LeftArrowKeyDown();
                        break;
                    case 114:
                        g_pInputManager->RightArrowKeyDown();
                        break;
                    case 111:
                        g_pInputManager->UpArrowKeyDown();
                        break;
                    case 116:
                        g_pInputManager->DownArrowKeyDown();
                        break;
                    case 27:
                        g_pInputManager->AsciiKeyDown('r');
                        break;
                }
                break;
            }
            case XCB_KEY_RELEASE: {
                auto key_code =
                    reinterpret_cast<xcb_key_release_event_t*>(pEvent)->detail;
                printf("[XcbApplication] Key Release: Keycode: %d\n", key_code);
                switch (key_code) {
                    case 113:
                        g_pInputManager->LeftArrowKeyUp();
                        break;
                    case 114:
                        g_pInputManager->RightArrowKeyUp();
                        break;
                    case 111:
                        g_pInputManager->UpArrowKeyUp();
                        break;
                    case 116:
                        g_pInputManager->DownArrowKeyUp();
                        break;
                    case 27:
                        g_pInputManager->AsciiKeyUp('r');
                        break;
                }
            } break;
            default:
                break;
        }
        free(pEvent);
    } else {
        if (xcb_connection_has_error(m_pConn)) {
            m_bQuit = true;
        }
    }
}
