#include "RhiComposition.hpp"

#include "config.h"

#if defined(OS_WINDOWS)
#include "Platform/Windows/OpenGLApplication.hpp"
#elif defined(OS_LINUX)
#include "Platform/Linux/OpenGLApplication.hpp"
#endif
#include "RHI/OpenGL/OpenGLGraphicsManager.hpp"
#include "RHI/OpenGL/OpenGLPipelineStateManager.hpp"

namespace My {

std::unique_ptr<RhiStartupComposition> ComposeOpenGLStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::OpenGL;
    // Render conventions follow the active session, not the build.
    composition->fixOpenGLPerspectiveMatrix = true;
    configuration.fixOpenGLPerspectiveMatrix = true;
    composition->application =
        std::make_unique<OpenGLApplication>(configuration);
    composition->pipelineStateManager =
        std::make_unique<OpenGLPipelineStateManager>();
    composition->graphicsManager = std::make_unique<OpenGLGraphicsManager>(
        sceneManager, *composition->pipelineStateManager);
    return composition;
}

}  // namespace My