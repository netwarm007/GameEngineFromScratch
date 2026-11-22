#include "SDLOpenGLBackend.hpp"

#include <cstdio>

#include "IApplication.hpp"
#include "IPlatformHost.hpp"
#include "imgui.h"
#include "imgui_impl_sdl2.h"

using namespace My;

SDLOpenGLBackend::SDLOpenGLBackend(IApplication& application,
                                   IPlatformHost& platform,
                                   std::function<SDL_Window*()> windowProvider,
                                   std::function<void(Uint32)> windowFlagsSetter)
    : m_Application(application),
      m_Platform(platform),
      m_WindowProvider(std::move(windowProvider)),
      m_WindowFlagsSetter(std::move(windowFlagsSetter)) {}

void SDLOpenGLBackend::CreateMainWindow() {
    // SDL_VideoInit resets GL attributes to defaults (2.1); initialize the
    // video subsystem before setting the requested context version.
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_InitSubSystem(VIDEO) error: %s\n", SDL_GetError());
        return;
    }
#if defined(OS_WEBASSEMBLY)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#endif
    // Xvfb exposes only RGB visuals without alpha, depth or multisample
    // GLX configs, and SDL defaults request alpha=8. The engine renders
    // into offscreen targets with their own depth/MSAA (GraphicsManager),
    // so the window framebuffer needs none of these; requesting them
    // makes SDL_CreateWindow fail with "Couldn't find matching GLX visual".
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    m_WindowFlagsSetter(SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    m_Application.CreateMainWindow();

    SDL_Window* window = m_WindowProvider();
    if (!window) {
        fprintf(stderr, "SDL OpenGL requires a native window\n");
        return;
    }
    m_hContext = SDL_GL_CreateContext(window);
    if (!m_hContext) {
        fprintf(stderr, "SDL_GL_CreateContext error: %s\n", SDL_GetError());
        return;
    }
    SDL_GL_MakeCurrent(window, m_hContext);
    SDL_GL_SetSwapInterval(1);  // Enable vsync

    int major_ver, minor_ver;
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major_ver);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor_ver);
    printf("Initialized GL context: %d.%d\n", major_ver, minor_ver);
}

int SDLOpenGLBackend::InitializeGraphicsResources() {
    if (!m_WindowProvider() || !m_hContext) {
        fprintf(stderr, "SDL OpenGL initialization requires a window and GL context\n");
        return -1;
    }
    return 0;
}

int SDLOpenGLBackend::InitializeGuiPlatform() {
    if (!ImGui::GetCurrentContext()) return 0;
    SDL_Window* window = m_WindowProvider();
    if (!window || !m_hContext || ImGui::GetIO().BackendPlatformUserData) {
        return -1;
    }
    if (!ImGui_ImplSDL2_InitForOpenGL(window, m_hContext)) return -1;
    return 0;
}

void SDLOpenGLBackend::FinalizeGuiPlatform() {
    if (ImGui::GetCurrentContext() && ImGui::GetIO().BackendPlatformUserData) {
        ImGui_ImplSDL2_Shutdown();
    }
}

void SDLOpenGLBackend::FinalizeGraphicsResources() {
    if (m_hContext) {
        SDL_GL_DeleteContext(m_hContext);
        m_hContext = nullptr;
    }
}

void SDLOpenGLBackend::PresentFrame() {
    if (SDL_Window* window = m_WindowProvider()) {
        SDL_GL_SwapWindow(window);
    }
}