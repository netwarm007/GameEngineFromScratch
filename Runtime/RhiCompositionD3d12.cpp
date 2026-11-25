#include "RhiComposition.hpp"

#include "Platform/Windows/D3d12Backend.hpp"
#include "Platform/Windows/WindowsApplication.hpp"
#include "RHI/D3d/D3d12GraphicsManager.hpp"
#include "RHI/D3d/D3d12PipelineStateManager.hpp"

namespace My {

RhiSessionModules ComposeD3d12Session(IRenderBackend& backendObject,
                                      ISceneManager& sceneManager,
                                      std::function<MemorySystem*()> memoryProvider) {
    auto& backend = static_cast<D3d12Backend&>(backendObject);
    RhiSessionModules modules;
    modules.pipelineStateManager = std::make_unique<D3d12PipelineStateManager>();
    modules.graphicsManager = std::make_unique<D3d12GraphicsManager>(
        backend.GetRHI(), sceneManager, *modules.pipelineStateManager,
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
    auto host = std::make_unique<WindowsApplication>(configuration);
    composition->sessionBackend =
        std::make_unique<D3d12Backend>(*host, *host);
    composition->recreateWindowOnReconstruction =
        composition->sessionBackend->RequiresWindowRecreation();
    if (!memoryProvider) {
        memoryProvider = [application = host.get()] {
            return application->GetMemorySystem();
        };
    }
    auto modules = ComposeD3d12Session(*composition->sessionBackend,
                                       sceneManager, std::move(memoryProvider));
    composition->application = std::move(host);
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

}  // namespace My
