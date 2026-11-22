#include "RhiComposition.hpp"

#include "Platform/Windows/D3d12Application.hpp"
#include "RHI/D3d/D3d12GraphicsManager.hpp"
#include "RHI/D3d/D3d12PipelineStateManager.hpp"

namespace My {

RhiSessionModules ComposeD3d12Session(BaseApplication& host,
                                      ISceneManager& sceneManager,
                                      std::function<MemorySystem*()> memoryProvider) {
    auto& application = static_cast<D3d12Application&>(host);
    RhiSessionModules modules;
    modules.pipelineStateManager = std::make_unique<D3d12PipelineStateManager>();
    modules.graphicsManager = std::make_unique<D3d12GraphicsManager>(
        application.GetRHI(), sceneManager, *modules.pipelineStateManager,
        std::move(memoryProvider));
    return modules;
}

std::unique_ptr<RhiStartupComposition> ComposeD3d12Startup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& /*assetReader*/,
    std::function<MemorySystem*()> memoryProvider) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::D3d12;
    configuration.fixOpenGLPerspectiveMatrix = false;
    composition->application =
        std::make_unique<D3d12Application>(configuration);
    if (!memoryProvider) {
        memoryProvider = [application = composition->application.get()] {
            return application->GetMemorySystem();
        };
    }
    auto modules = ComposeD3d12Session(*composition->application, sceneManager,
                                       std::move(memoryProvider));
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

}  // namespace My