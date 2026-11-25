#include <string.h>

#include "IRenderBackend.hpp"
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

    // GLX framebuffer configurations belong to the display that queried them.
    if (!m_pDisplay) m_pDisplay = XOpenDisplay(NULL);
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
    const uint16_t createdWidth =
        m_HasPreservedGeometry ? m_PreservedW : m_Config.screenWidth;
    const uint16_t createdHeight =
        m_HasPreservedGeometry ? m_PreservedH : m_Config.screenHeight;
    xcb_create_window(m_pConn,                       /* connection */
                      XCB_COPY_FROM_PARENT,          /* depth */
                      m_XWindow,                     /* window ID */
                      m_pScreen->root,               /* parent window */
                      m_PreservedX, m_PreservedY,    /* x, y */
                      createdWidth, createdHeight,
                      0,                             /* border width (X11
                         legacy; modern WMs draw their own decorations and a
                         non-zero border offsets the client origin from the
                         ICCCM-hinted position by the border under
                         StaticGravity. Keep 0: hints, create and capture are
                         then the same point. */
                      XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class */
                      m_pScreen->root_visual,        /* visual */
                      mask, values);                 /* masks */

    // A recreated (RHI-switch/recovery) window must keep the placement the
    // user left it in. WMs ignore create-window coordinates unless the ICCCM
    // WM_NORMAL_HINTS property declares a user-specified position/size, so
    // publish it before mapping. StaticGravity pins the client origin to the
    // requested root coordinates; the WM frame then grows around it, which is
    // exactly how the capture in FinalizePlatformResources reads it back.
    if (m_HasPreservedGeometry) {
        auto normalHints = xcb_intern_atom_reply(
            m_pConn, xcb_intern_atom(m_pConn, 0, 15, "WM_NORMAL_HINTS"), NULL);
        auto sizeHints = xcb_intern_atom_reply(
            m_pConn, xcb_intern_atom(m_pConn, 0, 13, "WM_SIZE_HINTS"), NULL);
        if (normalHints && sizeHints) {
            // WM_SIZE_HINTS layout (ICCCM 4.1.2.4): flags, x, y, width,
            // height, min/max size, resize increments, aspect ratios, base
            // size, gravity.
            uint32_t hints[18] = {0};
            hints[0] = 1 /* USPosition */ | 2 /* USSize */ |
                       512 /* PWinGravity */;
            hints[1] = static_cast<uint32_t>(m_PreservedX);
            hints[2] = static_cast<uint32_t>(m_PreservedY);
            hints[3] = createdWidth;
            hints[4] = createdHeight;
            hints[17] = XCB_GRAVITY_STATIC;
            xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_XWindow,
                                normalHints->atom, sizeHints->atom, 32, 18,
                                hints);
        }
        free(normalHints);
        free(sizeHints);
    }

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
}

int XcbApplication::InitializeGuiPlatform() {
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.BackendPlatformName) return -1;
        io.BackendPlatformName = "imgui_impl_x11";
        io.DisplaySize.x = static_cast<float>(m_Config.screenWidth);
        io.DisplaySize.y = static_cast<float>(m_Config.screenHeight);
        io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
    }
    return 0;
}

void XcbApplication::FinalizeGuiPlatform() {
    if (ImGui::GetCurrentContext()) {
        ImGui::GetIO().BackendPlatformName = nullptr;
    }
}

void XcbApplication::FinalizePlatformResources() {
    if (m_pConn) {
        if (m_XWindow != XCB_WINDOW_NONE) {
            auto geom = xcb_get_geometry_reply(
                m_pConn, xcb_get_geometry(m_pConn, m_XWindow), NULL);
            if (geom) {
                // geom->x/y are relative to the parent window: under a
                // reparenting WM that is the frame, so copying them would
                // bake the decoration offset into the next placement. Record
                // the client origin in root (screen) coordinates instead.
                int absX = geom->x, absY = geom->y;
                Window sibling = XCB_WINDOW_NONE;
                if (m_pDisplay &&
                    XTranslateCoordinates(m_pDisplay, m_XWindow,
                                          m_pScreen->root, 0, 0, &absX, &absY,
                                          &sibling)) {
                    m_PreservedX = static_cast<int16_t>(absX);
                    m_PreservedY = static_cast<int16_t>(absY);
                } else {
                    m_PreservedX = geom->x;
                    m_PreservedY = geom->y;
                }
                m_PreservedW = geom->width;
                m_PreservedH = geom->height;
                m_HasPreservedGeometry = true;
                free(geom);
            }
            xcb_destroy_window(m_pConn, m_XWindow);
            m_XWindow = XCB_WINDOW_NONE;
        }
        xcb_disconnect(m_pConn);
        m_pConn = nullptr;
        m_pScreen = nullptr;
    }
    if (m_pDisplay) {
        XCloseDisplay(m_pDisplay);
        m_pDisplay = nullptr;
    }
}

void XcbApplication::Tick() {
    ProcessEvents();
    BaseApplication::Tick();
}

void XcbApplication::ProcessEvents() {
    xcb_generic_event_t* pEvent;
    if ((pEvent = xcb_poll_for_event(m_pConn))) {
        switch (pEvent->response_type & ~0x80) {
            case XCB_EXPOSE: {
                XWindowAttributes gwa;
                XGetWindowAttributes(m_pDisplay, m_XWindow, &gwa);
                m_Config.screenWidth = gwa.width;
                m_Config.screenHeight = gwa.height;
                if (m_pActiveBackend) {
                    m_pActiveBackend->OnWindowResize(gwa.width, gwa.height);
                }
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

    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize.x = static_cast<float>(m_Config.screenWidth);
        io.DisplaySize.y = static_cast<float>(m_Config.screenHeight);
    }
}

void XcbApplication::GetFramebufferSize(uint32_t& width, uint32_t& height) {
    if (!m_pConn || m_XWindow == XCB_WINDOW_NONE) {
        width = height = 0;
        return;
    }
    auto geom = xcb_get_geometry_reply(
        m_pConn, xcb_get_geometry(m_pConn, m_XWindow), NULL);

    width = geom->width;
    height = geom->height;

    free(geom);
}

void XcbApplication::SetWindowTitle(const char* title) {
    if (!m_pConn || m_XWindow == XCB_WINDOW_NONE || !title) return;
    const size_t len = strlen(title);
    xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_XWindow,
                        XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, len, title);
    auto net = xcb_intern_atom_reply(
        m_pConn, xcb_intern_atom(m_pConn, 0, 12, "_NET_WM_NAME"), nullptr);
    auto utf8 = xcb_intern_atom_reply(
        m_pConn, xcb_intern_atom(m_pConn, 0, 11, "UTF8_STRING"), nullptr);
    if (net && utf8) {
        xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_XWindow,
                            net->atom, utf8->atom, 8, len, title);
    }
    free(net);
    free(utf8);
    xcb_flush(m_pConn);
}
