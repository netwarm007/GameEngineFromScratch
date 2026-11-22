#include "RhiComposition.hpp"

#include "Platform/Darwin/CocoaApplication.h"
#include "Platform/Darwin/MetalBackend.h"
#include "RHI/Metal/Metal2GraphicsManager.h"
#include "RHI/Metal/MetalPipelineStateManager.h"

namespace My {

RhiSessionModules ComposeMetalSession(IRenderBackend& backendObject,
                                      ISceneManager& sceneManager) {
    auto& backend = static_cast<MetalBackend&>(backendObject);
    RhiSessionModules modules;
    modules.pipelineStateManager = std::make_unique<MetalPipelineStateManager>();
    modules.graphicsManager = std::make_unique<Metal2GraphicsManager>(
        sceneManager, *modules.pipelineStateManager,
        [&backend] { return (void*)backend.GetMetalView(); });
    return modules;
}

std::unique_ptr<RhiStartupComposition> ComposeMetalStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::Metal;
    configuration.fixOpenGLPerspectiveMatrix = false;
    auto host = std::make_unique<CocoaApplication>(configuration);
    // Headless capture pacing matches the legacy Metal application.
    host->SetHeadlessPacing(true);
    composition->sessionBackend = std::make_unique<MetalBackend>(
        *host, *host, [host = host.get()] { return host->GetInputManager(); });
    composition->recreateWindowOnReconstruction =
        composition->sessionBackend->RequiresWindowRecreation();
    auto modules = ComposeMetalSession(*composition->sessionBackend, sceneManager);
    composition->application = std::move(host);
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

}  // namespace My
