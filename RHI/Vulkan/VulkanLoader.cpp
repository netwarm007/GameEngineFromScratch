#include "VulkanLoader.hpp"

#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace My {

VulkanLoader::VulkanLoader(const char* libraryPath) {
#ifdef _WIN32
    const auto module = libraryPath
                            ? LoadLibraryExA(libraryPath, nullptr,
                                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                                 LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
                            : LoadLibraryExA("vulkan-1.dll", nullptr,
                                             LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) {
        throw std::runtime_error("Cannot load Vulkan runtime: Windows error " +
                                 std::to_string(GetLastError()));
    }
    m_module = module;
#else
#ifdef __APPLE__
    const char* defaultLibrary = "libvulkan.1.dylib";
#else
    const char* defaultLibrary = "libvulkan.so.1";
#endif
    m_module = dlopen(libraryPath ? libraryPath : defaultLibrary, RTLD_NOW | RTLD_LOCAL);
    if (!m_module) {
        const auto* error = dlerror();
        throw std::runtime_error(
            "Cannot load Vulkan runtime: " +
            std::string(error ? error : "unknown dynamic loader error"));
    }
#endif
    try {
        m_getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetExportedFunction("vkGetInstanceProcAddr"));
        if (!m_getInstanceProcAddr) {
            throw std::runtime_error("Vulkan runtime is missing vkGetInstanceProcAddr");
        }
        for (const auto* name : {"vkCreateInstance",
                                 "vkEnumerateInstanceExtensionProperties",
                                 "vkEnumerateInstanceLayerProperties"}) {
            if (!GetInstanceFunction(VK_NULL_HANDLE, name)) {
                throw std::runtime_error(std::string("Vulkan runtime is missing ") + name);
            }
        }
    } catch (const std::exception&) {
        Close();
        throw;
    }
}

VulkanLoader::~VulkanLoader() { Close(); }

void VulkanLoader::Close() {
    if (!m_module) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(m_module));
#else
    dlclose(m_module);
#endif
    m_module = nullptr;
    m_getInstanceProcAddr = nullptr;
}

PFN_vkVoidFunction VulkanLoader::GetInstanceFunction(
    VkInstance instance, const char* name) const {
    if (!name || !*name) throw std::invalid_argument("Vulkan function name is empty");
    return m_getInstanceProcAddr(instance, name);
}

PFN_vkVoidFunction VulkanLoader::GetExportedFunction(const char* name) const {
    if (!name || !*name) throw std::invalid_argument("Vulkan function name is empty");
#ifdef _WIN32
    return reinterpret_cast<PFN_vkVoidFunction>(
        GetProcAddress(static_cast<HMODULE>(m_module), name));
#else
    return reinterpret_cast<PFN_vkVoidFunction>(dlsym(m_module, name));
#endif
}

VulkanLoader& GetVulkanLoader() {
    static VulkanLoader loader;
    return loader;
}

}  // namespace My
