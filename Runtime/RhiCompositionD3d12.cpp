#include "RhiComposition.hpp"

#include "Platform/Windows/D3d12Application.hpp"
#include "RHI/D3d/D3d12GraphicsManager.hpp"
#include "RHI/D3d/D3d12PipelineStateManager.hpp"

namespace My {

std::unique_ptr<RhiStartupComposition> ComposeD3d12Startup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::D3d12;
    configuration.fixOpenGLPerspectiveMatrix = false;
    composition->application =
        std::make_unique<D3d12Application>(configuration);
    composition->pipelineStateManager =
        std::make_unique<D3d12PipelineStateManager>();
    auto* application =
        static_cast<D3d12Application*>(composition->application.get());
    composition->graphicsManager = std::make_unique<D3d12GraphicsManager>(
        application->GetRHI(), sceneManager, *composition->pipelineStateManager,
        [application] { return application->GetMemorySystem(); });
    return composition;
}

}  // namespace My