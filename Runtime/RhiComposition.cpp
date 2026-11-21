#include "RhiComposition.hpp"

#include <utility>

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
    IAssetReader& assetReader);
#endif
#if defined(OS_WINDOWS) || defined(OS_LINUX)
std::unique_ptr<RhiStartupComposition> ComposeOpenGLStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader);
#endif
#if defined(OS_MACOS)
std::unique_ptr<RhiStartupComposition> ComposeMetalStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader);
#endif
#if defined(HAS_VULKAN)
std::unique_ptr<RhiStartupComposition> ComposeVulkanStartup(
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader);
RhiBackendAvailability ProbeVulkanRuntimeAvailability();
#endif

RhiCompositionResult CreateRhiStartupComposition(
    RhiBackend backend, GfxConfiguration& configuration,
    ISceneManager& sceneManager, IAssetReader& assetReader) {
    switch (backend) {
        case RhiBackend::D3d12:
#if defined(OS_WINDOWS)
            return {ComposeD3d12Startup(configuration, sceneManager, assetReader),
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