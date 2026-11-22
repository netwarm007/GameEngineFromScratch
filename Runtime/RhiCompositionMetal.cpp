#include "RhiComposition.hpp"

#include "Platform/Darwin/CocoaMetalApplication.h"
#include "RHI/Metal/Metal2GraphicsManager.h"
#include "RHI/Metal/MetalPipelineStateManager.h"

namespace My {

std::unique_ptr<RhiStartupComposition> ComposeMetalStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::Metal;
    configuration.fixOpenGLPerspectiveMatrix = false;
    composition->application =
        std::make_unique<CocoaMetalApplication>(configuration);
    composition->pipelineStateManager =
        std::make_unique<MetalPipelineStateManager>();
    auto* application =
        static_cast<CocoaMetalApplication*>(composition->application.get());
    composition->graphicsManager = std::make_unique<Metal2GraphicsManager>(
        sceneManager, *composition->pipelineStateManager,
        [application] { return application->GetMetalView(); });
    return composition;
}

}  // namespace My