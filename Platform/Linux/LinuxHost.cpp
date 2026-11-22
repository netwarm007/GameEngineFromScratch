#include "LinuxHost.hpp"

#include <cstring>
#include <iostream>

#include "IRenderBackend.hpp"
#include "InputManager.hpp"
#include "imgui.h"
#include "imgui_impl_sdl2.h"

using namespace My;
using namespace std;

LinuxHost::LinuxHost() = default;
LinuxHost::LinuxHost(GfxConfiguration& cfg) : BaseApplication(cfg) {}

int LinuxHost::Initialize() {
    const int result = BaseApplication::Initialize();
    if (result != 0) return result;

    if (m_Mode == Mode::Sdl && !m_bSdlInitialized) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            cerr << "SDL_Init error: " << SDL_GetError() << '\n';
            BaseApplication::Finalize();
            return 1;
        }
        m_bSdlInitialized = true;
    }
    return 0;
}

void LinuxHost::Finalize() {
    BaseApplication::Finalize();
    if (m_pWindow) {
        SDL_DestroyWindow(m_pWindow);
        m_pWindow = nullptr;
    }
    if (m_bSdlInitialized) {
        SDL_Quit();
        m_bSdlInitialized = false;
    }
}

void LinuxHost::Tick() {
    ProcessEvents();
    BaseApplication::Tick();
}

void LinuxHost::ProcessEvents() {
    if (m_Mode == Mode::Sdl) {
        ProcessSdlEvents();
    } else {
        ProcessXcbEvents();
    }
}

void LinuxHost::ProcessSdlEvents() {
    SDL_Event e;

    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e);

        switch (e.type) {
            case SDL_QUIT:
                m_bQuit = true;
                break;
            case SDL_KEYDOWN:
                break;
            case SDL_KEYUP: {
                if (m_pInputManager) {
                    m_pInputManager->AsciiKeyDown(
                        static_cast<char>(e.key.keysym.sym));
                }
            } break;
            case SDL_MOUSEBUTTONDOWN: {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    if (m_pInputManager) m_pInputManager->LeftMouseButtonDown();
                    m_bInDrag = true;
                }
            } break;
            case SDL_MOUSEBUTTONUP: {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    if (m_pInputManager) m_pInputManager->LeftMouseButtonUp();
                    m_bInDrag = false;
                }
            } break;
            case SDL_MOUSEMOTION: {
                if (m_bInDrag) {
                    if (m_pInputManager)
                        m_pInputManager->LeftMouseDrag(e.motion.xrel,
                                                       e.motion.yrel);
                }
            } break;
            case SDL_WINDOWEVENT:
                if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
                    int tmpX, tmpY;
                    SDL_GetWindowSize(m_pWindow, &tmpX, &tmpY);
                    onResize(tmpX, tmpY);
                }
        }
    }

    if (ImGui::GetCurrentContext()) {
        ImGui_ImplSDL2_NewFrame();
    }
}

void LinuxHost::ProcessXcbEvents() {
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

void LinuxHost::onResize(int width, int height) {
    if (height == 0) {
        height = 1;
    }
    if (m_pGraphicsManager) m_pGraphicsManager->ResizeCanvas(width, height);
    if (m_pActiveBackend) m_pActiveBackend->OnWindowResize(width, height);
}

void* LinuxHost::GetMainWindowHandler() {
    if (m_Mode == Mode::Sdl) return m_pWindow;
    return reinterpret_cast<void*>(m_XWindow);
}

void LinuxHost::CreateMainWindow() {
    if (m_Mode == Mode::Sdl) {
        CreateSdlWindow();
    } else {
        CreateXcbWindow();
    }
}

void LinuxHost::CreateSdlWindow() {
    m_pWindow = SDL_CreateWindow(m_Config.appName, SDL_WINDOWPOS_CENTERED,
                                 SDL_WINDOWPOS_CENTERED, m_Config.screenWidth,
                                 m_Config.screenHeight, m_WindowFlags);
    if (!m_pWindow) {
        cerr << "SDL_CreateWindow error: " << SDL_GetError() << '\n';
        return;
    }
    // SDL_CreateWindow implicitly initializes the video subsystem.
    m_bSdlInitialized = true;
}

void LinuxHost::CreateXcbWindow() {
    uint32_t mask = 0;
    uint32_t values[3];

    if (!m_pDisplay) m_pDisplay = XOpenDisplay(NULL);
    if (!m_pDisplay) {
        fprintf(stderr, "Can't open display\n");
        return;
    }

    if (!m_pConn) {
        m_pConn = xcb_connect(NULL, NULL);
    }

    if (!m_pScreen) {
        m_pScreen = xcb_setup_roots_iterator(xcb_get_setup(m_pConn)).data;
    }

    XSetEventQueueOwner(m_pDisplay, XCBOwnsEventQueue);

    xcb_window_t root = m_pScreen->root;

    xcb_colormap_t colormap = xcb_generate_id(m_pConn);
    xcb_create_colormap(m_pConn, XCB_COLORMAP_ALLOC_NONE, colormap, root,
                        m_pScreen->root_visual);

    auto foreground = xcb_generate_id(m_pConn);
    mask = XCB_GC_FOREGROUND | XCB_GC_GRAPHICS_EXPOSURES;
    values[0] = m_pScreen->black_pixel;
    values[1] = 0;
    xcb_create_gc(m_pConn, foreground, root, mask, values);

    auto background = xcb_generate_id(m_pConn);
    mask = XCB_GC_BACKGROUND | XCB_GC_GRAPHICS_EXPOSURES;
    values[0] = m_pScreen->white_pixel;
    values[1] = 0;
    xcb_create_gc(m_pConn, background, root, mask, values);

    m_XWindow = xcb_generate_id(m_pConn);
    mask = XCB_CW_EVENT_MASK | XCB_CW_COLORMAP;
    values[0] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;
    values[1] = colormap;
    values[2] = 0;
    xcb_create_window(m_pConn, XCB_COPY_FROM_PARENT, m_XWindow, root, 20, 20,
                      m_Config.screenWidth, m_Config.screenHeight, 10,
                      XCB_WINDOW_CLASS_INPUT_OUTPUT, m_pScreen->root_visual,
                      mask, values);

    xcb_change_property(m_pConn, XCB_PROP_MODE_REPLACE, m_XWindow,
                        XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
                        strlen(m_Config.appName), m_Config.appName);

    xcb_map_window(m_pConn, m_XWindow);
    xcb_flush(m_pConn);
}

void LinuxHost::GetFramebufferSize(uint32_t& width, uint32_t& height) {
    if (m_Mode == Mode::Sdl) {
        int w = 0, h = 0;
        if (m_pWindow) SDL_GetWindowSize(m_pWindow, &w, &h);
        width = static_cast<uint32_t>(w);
        height = static_cast<uint32_t>(h);
        return;
    }
    auto geom = xcb_get_geometry_reply(
        m_pConn, xcb_get_geometry(m_pConn, m_XWindow), NULL);
    if (geom) {
        width = geom->width;
        height = geom->height;
        free(geom);
    } else {
        width = height = 0;
    }
}

int LinuxHost::InitializeGuiPlatform() {
    if (m_Mode == Mode::Sdl) {
        // The SDL OpenGL backend binds ImGui_ImplSDL2 itself.
        return 0;
    }
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

void LinuxHost::FinalizeGuiPlatform() {
    if (m_Mode == Mode::Sdl) {
        // The SDL OpenGL backend unbinds ImGui_ImplSDL2 itself.
        return;
    }
    if (ImGui::GetCurrentContext()) {
        ImGui::GetIO().BackendPlatformName = nullptr;
    }
}

void LinuxHost::FinalizePlatformResources() {
    if (m_Mode == Mode::Sdl) {
        if (m_pWindow) {
            SDL_DestroyWindow(m_pWindow);
            m_pWindow = nullptr;
        }
        return;
    }
    if (m_pConn) {
        if (m_XWindow != XCB_WINDOW_NONE) {
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

void LinuxHost::ShutdownPlatform() {
    if (m_bSdlInitialized) {
        SDL_Quit();
        m_bSdlInitialized = false;
    }
}