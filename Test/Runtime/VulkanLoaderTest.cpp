#include "RhiStartup.hpp"
#include "VulkanLoader.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

std::string RejectLoader(const char* path, const char* diagnostic) {
    try {
        VulkanLoader loader(path);
    } catch (const std::runtime_error& error) {
        Expect(std::string(error.what()).find(diagnostic) != std::string::npos,
               "retain the loader failure cause");
        return error.what();
    }
    Expect(false, "reject an absent or incomplete Vulkan loader");
    return {};
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "VulkanLoaderTest requires three fixture libraries\n";
        return 2;
    }
    const std::string missing = std::string(argv[1]) + ".not-found";
    const auto missingReason = RejectLoader(missing.c_str(), "Cannot load Vulkan runtime");
    const std::array availability = {
        RhiBackendAvailability{RhiBackend::OpenGL,
                               RhiAvailability::ReadyForInitialization, {}},
        RhiBackendAvailability{RhiBackend::Vulkan,
                               RhiAvailability::RuntimeUnavailable, missingReason}};
    const auto defaultChoice =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::OpenGL, availability);
    const auto explicitChoice =
        ResolveRhiStartupBackend(RhiBackend::Vulkan, RhiBackend::OpenGL, availability);
    Expect(defaultChoice.backend == RhiBackend::OpenGL && !explicitChoice.backend &&
               explicitChoice.error.find(missingReason) != std::string::npos,
           "missing Vulkan runtime preserves another backend's selection, not Vulkan fallback");

    for (int attempt = 0; attempt < 3; ++attempt) {
        RejectLoader(argv[2], "vkGetInstanceProcAddr");
        RejectLoader(argv[3], "vkCreateInstance");
        VulkanLoader loader(argv[1]);
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            loader.GetInstanceFunction(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties"));
        uint32_t count = 1;
        Expect(enumerate && enumerate(&count, nullptr) == VK_SUCCESS && count == 0,
               "load and call a typed function after failed loads and teardown");
        Expect(!loader.GetInstanceFunction(VK_NULL_HANDLE, "vkUnknownFixtureFunction"),
               "missing functions are not reported as available");
        Expect(!loader.GetInstanceFunction(VK_NULL_HANDLE, "vkDestroySurfaceKHR") &&
                   loader.GetExportedFunction("vkDestroySurfaceKHR"),
               "resolve exported WSI trampolines without enabling an offscreen extension");
    }
    if (failures) return 1;
    std::cout << "Vulkan loader tests passed without a Vulkan driver\n";
    return 0;
}
