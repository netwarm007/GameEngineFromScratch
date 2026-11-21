#include <cstring>
#include <vulkan/vulkan_core.h>

#ifdef _WIN32
#define FIXTURE_EXPORT extern "C" __declspec(dllexport)
#else
#define FIXTURE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance* instance) {
    *instance = VK_NULL_HANDLE;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
}

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char*, uint32_t* count, VkExtensionProperties*) {
    *count = 0;
    return VK_SUCCESS;
}

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t* count, VkLayerProperties*) {
    *count = 0;
    return VK_SUCCESS;
}

FIXTURE_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(
    VkInstance, VkSurfaceKHR, const VkAllocationCallbacks*) {}

#ifndef MYGE_FIXTURE_NO_SYMBOL
FIXTURE_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
    VkInstance, const char* name) {
#ifndef MYGE_FIXTURE_NO_CREATE
    if (std::strcmp(name, "vkCreateInstance") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateInstance);
#endif
    if (std::strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateInstanceExtensionProperties);
    if (std::strcmp(name, "vkEnumerateInstanceLayerProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateInstanceLayerProperties);
    return nullptr;
}
#endif
