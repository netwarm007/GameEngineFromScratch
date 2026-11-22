#include "RhiComposition.hpp"

#include "config.h"

#if defined(OS_WINDOWS)
#include "Platform/Windows/OpenGLApplication.hpp"
#elif defined(OS_LINUX) && defined(HAS_SDL2)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_LINUX)
#include "Platform/Linux/OpenGLApplication.hpp"
#endif
#include "RHI/OpenGL/OpenGLGraphicsManager.hpp"
#include "RHI/OpenGL/OpenGLPipelineStateManager.hpp"

namespace My {

RhiSessionModules ComposeOpenGLSession(BaseApplication& /*host*/,
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
    // WGL pixel format and GLX/SDL contexts are window-lifetime bound.
    composition->recreateWindowOnReconstruction = true;
    composition->application =
        std::make_unique<OpenGLApplication>(configuration);
    auto modules = ComposeOpenGLSession(*composition->application, sceneManager);
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

}  // namespace My