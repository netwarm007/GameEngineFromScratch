#include <string.h>

#include <cstdio>
#include <iostream>

#include "XcbApplication.hpp"

#include "InputManager.hpp"

#include "imgui.h"

using namespace My;
using namespace std;

void XcbApplication::CreateMainWindow() {
    uint32_t mask = 0;
    uint32_t values[3];

    /* Open Xlib Display */
    m_pDisplay = XOpenDisplay(NULL);
    if (!m_pDisplay) {
        fprintf(stderr, "Can't open display\n");
    }

    if (!m_pConn) {
        /* establish connection to X server */
        m_pConn = xcb_connect(NULL, NULL);
    }

    if (!m_pScreen) {
        /* get the first screen */
        m_pScreen = xcb_setup_roots_iterator(xcb_get_setup(m_pConn)).data;
    }

    /* Acquire event queue ownership */
    XSetEventQueueOwner(m_pDisplay, XCBOwnsEventQueue);

    /* get the root window */
    m_XWindow = m_pScreen->root;

    /* Create XID's for colormap */
    xcb_colormap_t colormap = xcb_generate_id(m_pConn);

    xcb_create_colormap(m_pConn, XCB_COLORMAP_ALLOC_NONE, colormap, m_XWindow,
                        m_pScreen->root_visual);

    /* create black (foreground) graphic context */
    auto foreground = xcb_generate_id(m_pConn);
    mask = XCB_GC_FOREGROUND | XCB_GC_GRAPHICS_EXPOSURES;
    values[0] = m_pScreen->black_pixel;
    values[1] = 0;
    xcb_create_gc(m_pConn, foreground, m_XWindow, mask, values);

    /* create white (background) graphic context */
    auto background = xcb_generate_id(m_pConn);
    mask = XCB_GC_BACKGROUND | XCB_GC_GRAPHICS_EXPOSURES;
    values[0] = m_pScreen->white_pixel;
    values[1] = 0;
    xcb_create_gc(m_pConn, background, m_XWindow, mask, values);

    /* create window */
    m_XWindow = xcb_generate_id(m_pConn);
    mask = XCB_CW_EVENT_MASK | XCB_CW_COLORMAP;
    values[0] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;
    values[1] = colormap;
    values[2] = 0;
    xcb_create_window(m_pConn,                       /* connection */
                      XCB_COPY_FROM_PARENT,          /* depth */
                      m_XWindow,                     /* window ID */
                      m_pScreen->root,               /* parent window */
                      20, 20,                        /* x, y */
                      m_Config.screenWidth,          /* width */
                      m_Config.screenHeight,         /* height */
                      10,                            /* boarder width */
                      XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class */
                      m_pScreen->root_visual,        /* visual */
                      mask, values);                 /* masks */

    /* set the title of the window */
    xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_XWindow,
                        XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
                        strlen(m_Config.appName), m_Config.appName);

    /* set the title of the window icon */
    /*
    xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_Window,
                XCB_ATOM_WM_ICON_NAME, XCB_ATOM_STRING, 8,
                strlen(m_Config.appName), m_Config.appName);
    */

    /* map the window on the screen */
    xcb_map_window(m_pConn, m_XWindow);

    xcb_flush(m_pConn);

    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        io.BackendPlatformName = "imgui_impl_x11";
        io.DisplaySize.x = DisplayWidth(m_pDisplay, m_nScreen);
        io.DisplaySize.y = DisplayHeight(m_pDisplay, m_nScreen);
        io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
    }
}

void XcbApplication::Finalize() {
    xcb_destroy_window(m_pConn, m_XWindow);
    xcb_disconnect(m_pConn);
}

void XcbApplication::Tick() {
    xcb_generic_event_t* pEvent;
    if ((pEvent = xcb_poll_for_event(m_pConn))) {
        switch (pEvent->response_type & ~0x80) {
            case XCB_EXPOSE: {
                XWindowAttributes gwa;
                XGetWindowAttributes(m_pDisplay, m_XWindow, &gwa);
                m_Config.screenWidth = gwa.width;
                m_Config.screenHeight = gwa.height;
                break;
            }
            case XCB_KEY_PRESS: {
                auto key_code =
                    reinterpret_cast<xcb_key_press_event_t*>(pEvent)->detail;
                printf("[XcbApplication] Key Press: Keycode: %d\n", key_code);
                if (m_pInputManager) {
                    switch (key_code) {
                        case 113:
                            m_pInputManager->LeftArrowKeyDown();
                            break;
                        case 114:
                            m_pInputManager->RightArrowKeyDown();
                            break;
                        case 111:
                            m_pInputManager->UpArrowKeyDown();
                            break;
                        case 116:
                            m_pInputManager->DownArrowKeyDown();
                            break;
                        case 27:
                            m_pInputManager->AsciiKeyDown('r');
                            break;
                    }
                }
                break;
            }
            case XCB_KEY_RELEASE: {
                auto key_code =
                    reinterpret_cast<xcb_key_release_event_t*>(pEvent)->detail;
                printf("[XcbApplication] Key Release: Keycode: %d\n", key_code);
                if (m_pInputManager) {
                    switch (key_code) {
                        case 113:
                            m_pInputManager->LeftArrowKeyUp();
                            break;
                        case 114:
                            m_pInputManager->RightArrowKeyUp();
                            break;
                        case 111:
                            m_pInputManager->UpArrowKeyUp();
                            break;
                        case 116:
                            m_pInputManager->DownArrowKeyUp();
                            break;
                        case 27:
                            m_pInputManager->AsciiKeyUp('r');
                            break;
                    }
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

    BaseApplication::Tick();
}

void XcbApplication::GetFramebufferSize(int& width, int& height) {
    auto geom = xcb_get_geometry_reply(
        m_pConn, xcb_get_geometry(m_pConn, m_XWindow), NULL);

    width = geom->width;
    height = geom->height;

    free(geom);
}
