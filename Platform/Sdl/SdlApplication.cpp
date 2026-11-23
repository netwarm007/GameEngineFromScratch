#include "SdlApplication.hpp"

#include "IRenderBackend.hpp"
#include <iostream>
#include <utility>

#include "GraphicsManager.hpp"
#include "InputManager.hpp"

#include "imgui_impl_sdl2.h"

/*
 * Recurse through the list of arguments to clean up, cleaning up
 * the first one in the list each iteration.
 */
template <typename T, typename... Args>
void cleanup(T *t, Args &&...args) {
    // Cleanup the first item in the list
    cleanup(t);
    // Recurse to clean up the remaining arguments
    cleanup(std::forward<Args>(args)...);
}
/*
 * These specializations serve to free the passed argument and also provide the
 * base cases for the recursive call above, eg. when args is only a single item
 * one of the specializations below will be called by
 * cleanup(std::forward<Args>(args)...), ending the recursion
 * We also make it safe to pass nullptrs to handle situations where we
 * don't want to bother finding out which values failed to load (and thus are
 * null) but rather just want to clean everything up and let cleanup sort it out
 */
template <>
inline void cleanup<SDL_Window>(SDL_Window *win) {
    if (!win) {
        return;
    }
    SDL_DestroyWindow(win);
}
template <>
inline void cleanup<SDL_Renderer>(SDL_Renderer *ren) {
    if (!ren) {
        return;
    }
    SDL_DestroyRenderer(ren);
}
template <>
inline void cleanup<SDL_Texture>(SDL_Texture *tex) {
    if (!tex) {
        return;
    }
    SDL_DestroyTexture(tex);
}
template <>
inline void cleanup<SDL_Surface>(SDL_Surface *surf) {
    if (!surf) {
        return;
    }
    SDL_FreeSurface(surf);
}

using namespace std;
using namespace My;

/**
 * Log an SDL error with some error message to the output stream of our choice
 * @param os The output stream to write the message to
 * @param msg The error message to write, format will be msg error:
 * SDL_GetError()
 */
void SdlApplication::logSDLError(std::ostream &os, const std::string &msg) {
    os << msg << " error: " << SDL_GetError() << std::endl;
}

void SdlApplication::onResize(int width, int height) {
    if (height == 0) {
        height = 1;
    }

    if (m_pGraphicsManager) m_pGraphicsManager->ResizeCanvas(width, height);
    if (m_pActiveBackend) m_pActiveBackend->OnWindowResize(width, height);
}

int SdlApplication::Initialize() {
    const int result = BaseApplication::Initialize();
    if (result != 0) return result;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        logSDLError(std::cout, "SDL_Init");
        BaseApplication::Finalize();
        return 1;
    }

    return 0;
}

void SdlApplication::Finalize() {
    // GUI platform shutdown is owned by the backend's FinalizeGuiPlatform
    // hook, which BaseApplication::Finalize runs before graphics teardown.
    BaseApplication::Finalize();

    cleanup(m_pWindow);
    SDL_Quit();
}

void SdlApplication::FinalizePlatformResources() {
    if (m_pWindow) {
        SDL_GetWindowPosition(m_pWindow, &m_PreservedX, &m_PreservedY);
        SDL_GetWindowSize(m_pWindow, &m_PreservedW, &m_PreservedH);
        m_HasPreservedGeometry = true;
    }
    cleanup(m_pWindow);
    m_pWindow = nullptr;
    // Discard queued events for the retiring window.
    SDL_PumpEvents();
    SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);
}

void SdlApplication::ShutdownPlatform() { SDL_Quit(); }

void SdlApplication::Tick() {
    ProcessEvents();
    BaseApplication::Tick();
}

void SdlApplication::ProcessEvents() {
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

void SdlApplication::CreateMainWindow() {
    m_pWindow = SDL_CreateWindow(
        m_Config.appName,
        m_HasPreservedGeometry ? m_PreservedX : SDL_WINDOWPOS_CENTERED,
        m_HasPreservedGeometry ? m_PreservedY : SDL_WINDOWPOS_CENTERED,
        m_HasPreservedGeometry ? m_PreservedW : m_Config.screenWidth,
        m_HasPreservedGeometry ? m_PreservedH : m_Config.screenHeight,
        m_WindowFlags);
    if (!m_pWindow) {
        logSDLError(cout, "CreateMainWindow");
        SDL_Quit();
    }
}
