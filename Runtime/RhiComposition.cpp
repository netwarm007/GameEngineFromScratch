#include "RhiComposition.hpp"

#include <iostream>
#include <utility>

#include "Framework/Manager/RuntimeRoot.hpp"
#include "config.h"

namespace My {
namespace {
std::string NotCompiledError(RhiBackend backend) {
    return "Requested RHI '" + std::string(RhiBackendName(backend)) +
           "' is not compiled into this application";
}
}  // namespace

#if defined(OS_WINDOWS)
std::unique_ptr<RhiStartupComposition> ComposeD3d12Startup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader, std::function<MemorySystem*()> memoryProvider);
RhiSessionModules ComposeD3d12Session(
    BaseApplication& host, ISceneManager& sceneManager,
    std::function<MemorySystem*()> memoryProvider);
#endif
#if defined(OS_WINDOWS) || defined(OS_LINUX)
std::unique_ptr<RhiStartupComposition> ComposeOpenGLStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader);
RhiSessionModules ComposeOpenGLSession(BaseApplication& host,
                                       ISceneManager& sceneManager);
#endif
#if defined(OS_MACOS)
std::unique_ptr<RhiStartupComposition> ComposeMetalStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader);
RhiSessionModules ComposeMetalSession(BaseApplication& host,
                                      ISceneManager& sceneManager);
#endif
#if defined(HAS_VULKAN)
std::unique_ptr<RhiStartupComposition> ComposeVulkanStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader);
RhiSessionModules ComposeVulkanSession(BaseApplication& host,
                                       ISceneManager& sceneManager,
                                       IAssetReader& assetReader);
RhiBackendAvailability ProbeVulkanRuntimeAvailability();
#endif

RhiCompositionResult CreateRhiStartupComposition(
    RhiBackend backend, GfxConfiguration& configuration,
    ISceneManager& sceneManager, IAssetReader& assetReader,
    std::function<MemorySystem*()> memoryProvider) {
    switch (backend) {
        case RhiBackend::D3d12:
#if defined(OS_WINDOWS)
            return {ComposeD3d12Startup(configuration, sceneManager, assetReader,
                                        std::move(memoryProvider)),
                    {}};
#else
            return {nullptr, NotCompiledError(backend)};
#endif
        case RhiBackend::OpenGL:
#if defined(OS_WINDOWS) || defined(OS_LINUX)
            return {ComposeOpenGLStartup(configuration, sceneManager, assetReader),
                    {}};
#else
            return {nullptr, NotCompiledError(backend)};
#endif
        case RhiBackend::Metal:
#if defined(OS_MACOS)
            return {ComposeMetalStartup(configuration, sceneManager, assetReader),
                    {}};
#else
            return {nullptr, NotCompiledError(backend)};
#endif
        case RhiBackend::Vulkan:
#if defined(HAS_VULKAN)
            return {ComposeVulkanStartup(configuration, sceneManager, assetReader),
                    {}};
#else
            return {nullptr, NotCompiledError(backend)};
#endif
    }
    return {nullptr, "Invalid RHI backend"};
}

RhiSessionResult CreateRhiSessionModules(
    RhiBackend backend, BaseApplication& host, ISceneManager& sceneManager,
    IAssetReader& assetReader, std::function<MemorySystem*()> memoryProvider) {
    switch (backend) {
        case RhiBackend::D3d12:
#if defined(OS_WINDOWS)
            return {ComposeD3d12Session(host, sceneManager,
                                        std::move(memoryProvider)),
                    {}};
#else
            return {{}, NotCompiledError(backend)};
#endif
        case RhiBackend::OpenGL:
#if defined(OS_WINDOWS) || defined(OS_LINUX)
            return {ComposeOpenGLSession(host, sceneManager), {}};
#else
            return {{}, NotCompiledError(backend)};
#endif
        case RhiBackend::Metal:
#if defined(OS_MACOS)
            return {ComposeMetalSession(host, sceneManager), {}};
#else
            return {{}, NotCompiledError(backend)};
#endif
        case RhiBackend::Vulkan:
#if defined(HAS_VULKAN)
            return {ComposeVulkanSession(host, sceneManager, assetReader), {}};
#else
            return {{}, NotCompiledError(backend)};
#endif
    }
    return {{}, "Invalid RHI backend"};
}

int ReconstructRhiSession(RuntimeRoot& root,
                          RhiStartupComposition& composition,
                          ISceneManager& sceneManager, IAssetReader& assetReader,
                          std::function<MemorySystem*()> memoryProvider) {
    if (!composition.application || !root.Backend()) return -1;
    const bool recreateWindow = composition.recreateWindowOnReconstruction;

    // Retire session modules while they are still alive, then release them.
    root.RetireSessionForReconstruction(recreateWindow);
    composition.graphicsManager.reset();
    composition.pipelineStateManager.reset();

    if (recreateWindow) composition.application->CreateMainWindow();
    const int prepared = root.PrepareReconstructedSession(recreateWindow);
    if (prepared != 0) return prepared;

    auto fresh = CreateRhiSessionModules(composition.backend,
                                         *composition.application, sceneManager,
                                         assetReader, std::move(memoryProvider));
    if (!fresh.modules.graphicsManager || !fresh.modules.pipelineStateManager) {
        std::cerr << "Session reconstruction failed: " << fresh.error << '\n';
        return -1;
    }
    composition.pipelineStateManager = std::move(fresh.modules.pipelineStateManager);
    composition.graphicsManager = std::move(fresh.modules.graphicsManager);
    root.RegisterSessionModule(*composition.graphicsManager);
    root.RegisterSessionModule(*composition.pipelineStateManager);
    const int activated = root.Session().Activate(
        [&root] { root.Backend()->WaitForGraphicsIdle(); });
    if (activated != 0) {
        composition.graphicsManager.reset();
        composition.pipelineStateManager.reset();
        return activated;
    }
    return 0;
}

std::vector<RhiBackendAvailability> ProbeRhiStartupAvailability() {
    std::vector<RhiBackendAvailability> availability;
#if defined(OS_WINDOWS)
    availability.push_back(
        {RhiBackend::D3d12, RhiAvailability::ReadyForInitialization, {}});
#else
    availability.push_back({RhiBackend::D3d12, RhiAvailability::NotCompiled,
                            "D3D12 is a Windows backend"});
#endif
#if defined(OS_WINDOWS) || defined(OS_LINUX)
    availability.push_back(
        {RhiBackend::OpenGL, RhiAvailability::ReadyForInitialization, {}});
#else
    availability.push_back({RhiBackend::OpenGL, RhiAvailability::NotCompiled,
                            "OpenGL native application is Windows/Linux only"});
#endif
#if defined(OS_MACOS)
    availability.push_back(
        {RhiBackend::Metal, RhiAvailability::ReadyForInitialization, {}});
#else
    availability.push_back({RhiBackend::Metal, RhiAvailability::NotCompiled,
                            "Metal is a macOS backend"});
#endif
#if defined(HAS_VULKAN)
    availability.push_back(ProbeVulkanRuntimeAvailability());
#else
    availability.push_back({RhiBackend::Vulkan, RhiAvailability::NotCompiled,
                            "Vulkan support was not configured"});
#endif
    return availability;
}

std::string_view RhiStartupPlatformName() {
#if defined(OS_WINDOWS)
    return "Windows";
#elif defined(OS_MACOS)
    return "macOS";
#elif defined(OS_LINUX)
    return "Linux";
#else
    return "Unknown";
#endif
}

std::string_view RhiStartupHostName(RhiBackend backend) {
#if defined(OS_LINUX)
    if (backend == RhiBackend::OpenGL) {
#if defined(HAS_SDL2)
        return "SDL";
#else
        return "XCB";
#endif
    }
    return "XCB";
#elif defined(OS_WINDOWS)
    return "Win32";
#elif defined(OS_MACOS)
    return "Cocoa";
#else
    return "native";
#endif
}

}  // namespace My