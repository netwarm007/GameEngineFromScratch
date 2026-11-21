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

std::unique_ptr<RhiStartupComposition> ComposeVulkanStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::Vulkan;
    configuration.fixOpenGLPerspectiveMatrix = false;
    auto application = std::make_unique<VulkanApplication>(configuration);
    // The managers borrow the RHI owned by the application; the composition's
    // member order destroys them before the application releases the session.
    VulkanRHI& rhi = application->GetRHI();
    composition->application = std::move(application);
    auto pipelines =
        std::make_unique<VulkanPipelineStateManager>(rhi, assetReader);
    composition->graphicsManager = std::make_unique<VulkanGraphicsManager>(
        rhi, sceneManager, *pipelines);
    composition->pipelineStateManager = std::move(pipelines);
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