#include <cstring>
#include <vulkan/vulkan_core.h>

#ifdef _WIN32
#define FIXTURE_EXPORT extern "C" __declspec(dllexport)
#else
#define FIXTURE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#ifdef MYGE_FIXTURE_WSI_PROBE
// Controlled fake WSI runtime for headless probe tests. GEFS_FIXTURE_WSI_MODE
// selects what the fake runtime exposes; "khr" (the default) is a fully
// capable driver, the other modes knock out one native-presentation gate each.
#include <cstdlib>

namespace {
int instanceBalance = 0;

const char* fixtureMode() {
    const char* mode = std::getenv("GEFS_FIXTURE_WSI_MODE");
    return mode ? mode : "khr";
}

bool modeIs(const char* mode) { return std::strcmp(fixtureMode(), mode) == 0; }

VkResult fillExtensions(const char* const* names, uint32_t nameCount,
                        uint32_t* count, VkExtensionProperties* properties) {
    if (!properties) {
        *count = nameCount;
        return VK_SUCCESS;
    }
    const uint32_t written = *count < nameCount ? *count : nameCount;
    for (uint32_t i = 0; i < written; ++i) {
        std::strcpy(properties[i].extensionName, names[i]);
        properties[i].specVersion = 1;
    }
    *count = written;
    return written < nameCount ? VK_INCOMPLETE : VK_SUCCESS;
}
}  // namespace

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* info, const VkAllocationCallbacks*, VkInstance* instance) {
    // Emulate a portability-only ICD (MoltenVK): the loader hides it unless
    // the instance enables VK_KHR_portability_enumeration with the
    // enumerate-portability bit, exactly like the engine's Apple gate.
    if (modeIs("portability-required")) {
        bool enumerated = false;
#if defined(VK_KHR_portability_enumeration)
        if (info && (info->flags & VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)) {
            for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
                if (std::strcmp(info->ppEnabledExtensionNames[i],
                                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
                    enumerated = true;
                }
            }
        }
#endif
        if (!enumerated) return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    ++instanceBalance;
    *instance = reinterpret_cast<VkInstance>(static_cast<uintptr_t>(1));
    return VK_SUCCESS;
}

FIXTURE_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(
    VkInstance, const VkAllocationCallbacks*) {
    --instanceBalance;
}

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char*, uint32_t* count, VkExtensionProperties* properties) {
    // Offer only extension strings whose guards exist in the compiled headers:
    // KHR surface/swapchain-maintenance1 guards arrived after the EXT ones
    // (e.g. 1.4.315 headers expose only EXT), and the probe mirrors those
    // compile-time gates exactly like VulkanRHI does.
    const char* names[5];
    uint32_t nameCount = 0;
    names[nameCount++] = "VK_KHR_surface";
    if (!modeIs("no-cap2")) names[nameCount++] = "VK_KHR_get_surface_capabilities2";
    if (!modeIs("no-surface-maintenance")) {
        if (modeIs("ext")) {
#if defined(VK_EXT_swapchain_maintenance1)
            names[nameCount++] = "VK_EXT_surface_maintenance1";
#elif defined(VK_KHR_swapchain_maintenance1)
            names[nameCount++] = "VK_KHR_surface_maintenance1";
#endif
        } else {
#if defined(VK_KHR_swapchain_maintenance1)
            names[nameCount++] = "VK_KHR_surface_maintenance1";
#endif
#if defined(VK_EXT_swapchain_maintenance1)
            names[nameCount++] = "VK_EXT_surface_maintenance1";
#endif
        }
    }
#if defined(VK_KHR_portability_enumeration)
    if (modeIs("portability-required")) {
        names[nameCount++] = "VK_KHR_portability_enumeration";
    }
#endif
    return fillExtensions(names, nameCount, count, properties);
}

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t* count, VkLayerProperties*) {
    *count = 0;
    return VK_SUCCESS;
}

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(
    VkInstance, uint32_t* count, VkPhysicalDevice* devices) {
    if (!devices) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count == 0) return VK_INCOMPLETE;
    devices[0] = reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(2));
    *count = 1;
    return VK_SUCCESS;
}

FIXTURE_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice, const char*, uint32_t* count,
    VkExtensionProperties* properties) {
    const char* names[3];
    uint32_t nameCount = 0;
    names[nameCount++] = "VK_KHR_swapchain";
    if (!modeIs("no-swapchain-maintenance")) {
        if (modeIs("ext")) {
#if defined(VK_EXT_swapchain_maintenance1)
            names[nameCount++] = "VK_EXT_swapchain_maintenance1";
#elif defined(VK_KHR_swapchain_maintenance1)
            names[nameCount++] = "VK_KHR_swapchain_maintenance1";
#endif
        } else {
#if defined(VK_KHR_swapchain_maintenance1)
            names[nameCount++] = "VK_KHR_swapchain_maintenance1";
#endif
#if defined(VK_EXT_swapchain_maintenance1)
            names[nameCount++] = "VK_EXT_swapchain_maintenance1";
#endif
        }
    }
    return fillExtensions(names, nameCount, count, properties);
}

FIXTURE_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    const VkBool32 enabled = modeIs("feature-off") ? VK_FALSE : VK_TRUE;
    for (VkBaseOutStructure* next =
             reinterpret_cast<VkBaseOutStructure*>(features->pNext);
         next; next = next->pNext) {
#if defined(VK_KHR_swapchain_maintenance1)
        if (next->sType ==
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR) {
            reinterpret_cast<VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR*>(next)
                ->swapchainMaintenance1 = enabled;
        }
#endif
#if defined(VK_EXT_swapchain_maintenance1)
        if (next->sType ==
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT) {
            reinterpret_cast<VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT*>(next)
                ->swapchainMaintenance1 = enabled;
        }
#endif
    }
}

FIXTURE_EXPORT int GEFS_FixtureInstanceBalance() { return instanceBalance; }

FIXTURE_EXPORT int GEFS_FixtureWsiMacroStatus() {
    int status = 0;
#if defined(VK_KHR_swapchain_maintenance1)
    status |= 1;
#endif
#if defined(VK_EXT_swapchain_maintenance1)
    status |= 2;
#endif
    return status;
}

FIXTURE_EXPORT int GEFS_FixtureHeaderVersion() { return VK_HEADER_VERSION; }

FIXTURE_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
    VkInstance, const char* name) {
    if (std::strcmp(name, "vkCreateInstance") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateInstance);
    if (std::strcmp(name, "vkDestroyInstance") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyInstance);
    if (std::strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateInstanceExtensionProperties);
    if (std::strcmp(name, "vkEnumerateInstanceLayerProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateInstanceLayerProperties);
    if (std::strcmp(name, "vkEnumeratePhysicalDevices") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumeratePhysicalDevices);
    if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateDeviceExtensionProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceFeatures2);
    return nullptr;
}

#else  // !MYGE_FIXTURE_WSI_PROBE

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

#endif  // MYGE_FIXTURE_WSI_PROBE
