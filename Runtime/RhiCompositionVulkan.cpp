#include "RhiComposition.hpp"

#include <exception>
#include <utility>

#include "config.h"

#if defined(OS_WINDOWS)
#include "Platform/Windows/VulkanApplication.hpp"
#include "Platform/Windows/WindowsApplication.hpp"
#elif defined(OS_MACOS)
#include "Platform/Darwin/CocoaApplication.h"
#include "Platform/Darwin/VulkanApplication.hpp"
#elif defined(OS_LINUX)
#include "Platform/Linux/VulkanApplication.hpp"
#include "Platform/Linux/XcbApplication.hpp"
#if defined(HAS_SDL2)
#include "Platform/Linux/LinuxHost.hpp"
#endif
#endif
#include "RHI/Vulkan/VulkanGraphicsManager.hpp"
#include "RHI/Vulkan/VulkanLoader.hpp"
#include "RHI/Vulkan/VulkanPipelineStateManager.hpp"

namespace My {

RhiSessionModules ComposeVulkanSession(IRenderBackend& backendObject,
                                       ISceneManager& sceneManager,
                                       IAssetReader& assetReader) {
    auto& backend = static_cast<VulkanBackend&>(backendObject);
    VulkanRHI& rhi = backend.GetRHI();
    RhiSessionModules modules;
    // The Vulkan graphics manager borrows the concrete pipeline manager.
    auto pipelines =
        std::make_unique<VulkanPipelineStateManager>(rhi, assetReader);
    modules.graphicsManager = std::make_unique<VulkanGraphicsManager>(
        rhi, sceneManager, *pipelines);
    modules.pipelineStateManager = std::move(pipelines);
    return modules;
}

std::unique_ptr<RhiStartupComposition> ComposeVulkanStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader) {
    auto composition = std::make_unique<RhiStartupComposition>();
    composition->backend = RhiBackend::Vulkan;
    configuration.fixOpenGLPerspectiveMatrix = false;
#if defined(OS_WINDOWS)
    auto host = std::make_unique<WindowsApplication>(configuration);
    composition->sessionBackend = MakeWin32VulkanBackend(*host);
#elif defined(OS_MACOS)
    auto host = std::make_unique<CocoaApplication>(configuration);
    composition->sessionBackend = MakeCocoaVulkanBackend(*host);
#elif defined(OS_LINUX)
#if defined(HAS_SDL2)
    auto host = std::make_unique<LinuxHost>(configuration);
    host->SetMode(LinuxHost::Mode::Xcb);
    composition->sessionBackend =
        MakeXcbVulkanBackend(*host, *host, host->GetXcbConnection());
#else
    auto host = std::make_unique<XcbApplication>(configuration);
    composition->sessionBackend =
        MakeXcbVulkanBackend(*host, *host, host->GetXcbConnection());
#endif
#endif
    composition->recreateWindowOnReconstruction =
        composition->sessionBackend->RequiresWindowRecreation();
    auto modules = ComposeVulkanSession(*composition->sessionBackend,
                                        sceneManager, assetReader);
    composition->application = std::move(host);
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
