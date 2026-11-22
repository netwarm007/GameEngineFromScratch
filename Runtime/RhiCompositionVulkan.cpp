#include "RhiComposition.hpp"

#include <exception>
#include <utility>

#include "config.h"

#if defined(OS_WINDOWS)
#include "Platform/Windows/VulkanApplication.hpp"
#elif defined(OS_MACOS)
#include "Platform/Darwin/VulkanApplication.hpp"
#elif defined(OS_LINUX)
#include "Platform/Linux/VulkanApplication.hpp"
#endif
#include "RHI/Vulkan/VulkanGraphicsManager.hpp"
#include "RHI/Vulkan/VulkanLoader.hpp"
#include "RHI/Vulkan/VulkanPipelineStateManager.hpp"

namespace My {

RhiSessionModules ComposeVulkanSession(BaseApplication& host,
                                       ISceneManager& sceneManager,
                                       IAssetReader& assetReader) {
    auto& application = static_cast<VulkanApplication&>(host);
    VulkanRHI& rhi = application.GetRHI();
    RhiSessionModules modules;
    modules.pipelineStateManager =
        std::make_unique<VulkanPipelineStateManager>(rhi, assetReader);
    modules.graphicsManager = std::make_unique<VulkanGraphicsManager>(
        rhi, sceneManager, *modules.pipelineStateManager);
    return modules;
}

std::unique_ptr<RhiStartupComposition> ComposeVulkanStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::Vulkan;
    configuration.fixOpenGLPerspectiveMatrix = false;
    composition->application = std::make_unique<VulkanApplication>(configuration);
    // The managers borrow the RHI owned by the persistent host application;
    // the composition's member order destroys them before the application
    // releases the session.
    auto modules = ComposeVulkanSession(*composition->application, sceneManager,
                                        assetReader);
    composition->pipelineStateManager = std::move(modules.pipelineStateManager);
    composition->graphicsManager = std::move(modules.graphicsManager);
    return composition;
}

RhiBackendAvailability ProbeVulkanRuntimeAvailability() {
    try {
        GetVulkanLoader();
        return {RhiBackend::Vulkan, RhiAvailability::ReadyForInitialization, {}};
    } catch (const std::exception& error) {
        return {RhiBackend::Vulkan, RhiAvailability::RuntimeUnavailable,
                error.what()};
    }
}

}  // namespace My