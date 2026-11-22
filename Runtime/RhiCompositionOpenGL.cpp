#include "RhiComposition.hpp"

#include "config.h"

#if defined(OS_WINDOWS)
#include "Platform/Windows/Win32OpenGLBackend.hpp"
#include "Platform/Windows/WindowsApplication.hpp"
#elif defined(OS_LINUX) && defined(HAS_SDL2)
#include "Platform/Linux/LinuxHost.hpp"
#include "Platform/Sdl/SDLOpenGLBackend.hpp"
#elif defined(OS_LINUX)
#include "Platform/Linux/OpenGLApplication.hpp"
#endif
#include "RHI/OpenGL/OpenGLGraphicsManager.hpp"
#include "RHI/OpenGL/OpenGLPipelineStateManager.hpp"

namespace My {

RhiSessionModules ComposeOpenGLSession(IRenderBackend& /*backendObject*/,
                                       ISceneManager& sceneManager) {
    RhiSessionModules modules;
    modules.pipelineStateManager = std::make_unique<OpenGLPipelineStateManager>();
    modules.graphicsManager = std::make_unique<OpenGLGraphicsManager>(
        sceneManager, *modules.pipelineStateManager);
    return modules;
}

std::unique_ptr<RhiStartupComposition> ComposeOpenGLStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::OpenGL;
    // Render conventions follow the active session, not the build.
    composition->fixOpenGLPerspectiveMatrix = true;
    configuration.fixOpenGLPerspectiveMatrix = true;
#if defined(OS_LINUX) && !defined(HAS_SDL2)
    // Legacy GLX fallback: the application implements the backend itself.
    composition->application = std::make_unique<OpenGLApplication>(configuration);
    composition->recreateWindowOnReconstruction = true;
#else
#if defined(OS_WINDOWS)
    auto host = std::make_unique<WindowsApplication>(configuration);
    composition->sessionBackend =
        std::make_unique<Win32OpenGLBackend>(*host, *host);
#else
    auto host = std::make_unique<LinuxHost>(configuration);
    host->SetMode(LinuxHost::Mode::Sdl);
    composition->sessionBackend = std::make_unique<SDLOpenGLBackend>(
        *host, *host, [host = host.get()] { return host->GetSdlWindow(); },
        [host = host.get()](Uint32 flags) { host->SetWindowFlags(flags); });
#endif
    composition->recreateWindowOnReconstruction =
        composition->sessionBackend->RequiresWindowRecreation();
    composition->application = std::move(host);
#endif
    IRenderBackend* backendObject = composition->sessionBackend.get();
    if (!backendObject) {
        backendObject =
            dynamic_cast<IRenderBackend*>(composition->application.get());
    }
    auto modules = ComposeOpenGLSession(*backendObject, sceneManager);
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

}  // namespace My
