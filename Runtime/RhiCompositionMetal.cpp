#include "RhiComposition.hpp"

#include "Platform/Darwin/CocoaMetalApplication.h"
#include "RHI/Metal/Metal2GraphicsManager.h"
#include "RHI/Metal/MetalPipelineStateManager.h"

namespace My {

RhiSessionModules ComposeMetalSession(BaseApplication& host,
                                      ISceneManager& sceneManager) {
    auto& application = static_cast<CocoaMetalApplication&>(host);
    RhiSessionModules modules;
    modules.pipelineStateManager = std::make_unique<MetalPipelineStateManager>();
    modules.graphicsManager = std::make_unique<Metal2GraphicsManager>(
        sceneManager, *modules.pipelineStateManager,
        [&application] { return (void*)application.GetMetalView(); });
    return modules;
}

std::unique_ptr<RhiStartupComposition> ComposeMetalStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::Metal;
    configuration.fixOpenGLPerspectiveMatrix = false;
    composition->application =
        std::make_unique<CocoaMetalApplication>(configuration);
    auto modules = ComposeMetalSession(*composition->application, sceneManager);
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

}  // namespace My