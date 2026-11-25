#include "RhiStartup.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "VulkanLoader.hpp"

namespace My {

RhiBackendAvailability ProbeVulkanWsiAvailability(VulkanLoader& loader) {
    const auto unavailable = [](std::string reason) -> RhiBackendAvailability {
        return {RhiBackend::Vulkan, RhiAvailability::RuntimeUnavailable,
                std::move(reason)};
    };

#if !defined(VK_KHR_swapchain_maintenance1) && !defined(VK_EXT_swapchain_maintenance1)
    return unavailable(
        "Vulkan SDK headers lack swapchain-maintenance1; native presentation "
        "is unsupported (headless Vulkan 1.1 remains supported)");
#else
    // The vulkan-hpp default dispatcher is only initialized inside a Vulkan
    // session; resolve raw entry points through the loader instead.
    const auto enumerateInstanceExtensions =
        reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            loader.GetInstanceFunction(VK_NULL_HANDLE,
                                       "vkEnumerateInstanceExtensionProperties"));
    const auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(
        loader.GetInstanceFunction(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!enumerateInstanceExtensions || !createInstance) {
        return unavailable(
            "Vulkan runtime is missing instance enumeration entry points");
    }

    std::vector<VkExtensionProperties> instanceExtensions;
    for (int attempt = 0; attempt < 3; ++attempt) {
        uint32_t count = 0;
        VkResult result = enumerateInstanceExtensions(nullptr, &count, nullptr);
        if (result != VK_SUCCESS) {
            return unavailable(
                "vkEnumerateInstanceExtensionProperties failed (VkResult " +
                std::to_string(result) + ")");
        }
        instanceExtensions.resize(count);
        result = enumerateInstanceExtensions(nullptr, &count,
                                             instanceExtensions.data());
        if (result == VK_SUCCESS) break;
        if (result != VK_INCOMPLETE) {
            return unavailable(
                "vkEnumerateInstanceExtensionProperties failed (VkResult " +
                std::to_string(result) + ")");
        }
    }
    const auto instanceHas = [&](const char* name) {
        return std::any_of(instanceExtensions.begin(), instanceExtensions.end(),
                           [name](const VkExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName, name) == 0;
                           });
    };

    // Mirror VulkanRHI::Initialize's native presentation gate: the instance
    // must expose get_surface_capabilities2 plus a surface-maintenance1
    // extension. These are not promoted to core (through Vulkan 1.4), so a
    // string check is the correct evidence.
    std::string missing;
    if (!instanceHas(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) {
        missing += " VK_KHR_get_surface_capabilities2";
    }
#ifdef VK_KHR_swapchain_maintenance1
    const bool khrSurface =
        instanceHas(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
#else
    const bool khrSurface = false;
#endif
#ifdef VK_EXT_swapchain_maintenance1
    const bool extSurface =
        instanceHas(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
#else
    const bool extSurface = false;
#endif
    if (!khrSurface && !extSurface) {
        missing += " VK_KHR/EXT_surface_maintenance1";
    }
    if (!missing.empty()) {
        return unavailable(
            "Native Vulkan presentation requires extensions this Vulkan "
            "runtime does not expose:" + missing);
    }

    // A minimal instance is enough to enumerate physical devices; no device,
    // surface, or window is created.
    VkApplicationInfo applicationInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                      nullptr, "GEFS Vulkan WSI probe", 0,
                                      "GEFS", 0, VK_API_VERSION_1_1};
    // Mirror the engine's Apple portability gate: without the enumerate-
    // portability bit the loader hides portability ICDs (MoltenVK), and
    // instance creation fails with VK_ERROR_INCOMPATIBLE_DRIVER.
    const char* enabledInstanceExtensions[1];
    uint32_t enabledInstanceExtensionCount = 0;
    VkInstanceCreateFlags instanceFlags = 0;
#if defined(__APPLE__) && defined(VK_KHR_portability_enumeration)
    if (instanceHas(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabledInstanceExtensions[enabledInstanceExtensionCount++] =
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        instanceFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
    VkInstanceCreateInfo instanceCreateInfo{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, instanceFlags,
        &applicationInfo, 0, nullptr, enabledInstanceExtensionCount,
        enabledInstanceExtensions};
    VkInstance instance = VK_NULL_HANDLE;
    const VkResult created = createInstance(&instanceCreateInfo, nullptr, &instance);
    if (created != VK_SUCCESS) {
        return unavailable("Vulkan instance creation failed (VkResult " +
                           std::to_string(created) + ")");
    }
    const auto destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        loader.GetInstanceFunction(instance, "vkDestroyInstance"));
    struct InstanceGuard {
        PFN_vkDestroyInstance destroy;
        VkInstance instance;
        ~InstanceGuard() {
            if (destroy) destroy(instance, nullptr);
        }
    } guard{destroyInstance, instance};

    const auto enumeratePhysicalDevices =
        reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
            loader.GetInstanceFunction(instance, "vkEnumeratePhysicalDevices"));
    const auto enumerateDeviceExtensions =
        reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            loader.GetInstanceFunction(instance,
                                       "vkEnumerateDeviceExtensionProperties"));
    const auto getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        loader.GetInstanceFunction(instance, "vkGetPhysicalDeviceFeatures2"));
    if (!enumeratePhysicalDevices || !enumerateDeviceExtensions || !getFeatures2) {
        return unavailable(
            "Vulkan runtime is missing device enumeration entry points");
    }

    uint32_t deviceCount = 0;
    VkResult result = enumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS) {
        return unavailable("vkEnumeratePhysicalDevices failed (VkResult " +
                           std::to_string(result) + ")");
    }
    if (deviceCount == 0) {
        return unavailable("No Vulkan device reported by the runtime");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    result = enumeratePhysicalDevices(instance, &deviceCount, devices.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        return unavailable("vkEnumeratePhysicalDevices failed (VkResult " +
                           std::to_string(result) + ")");
    }
    devices.resize(deviceCount);

    // Mirror findPresentFenceExtension: a swapchain-maintenance1 device
    // extension paired with the matching surface extension, with the
    // swapchainMaintenance1 feature enabled.
    for (VkPhysicalDevice device : devices) {
        uint32_t extensionCount = 0;
        if (enumerateDeviceExtensions(device, nullptr, &extensionCount,
                                      nullptr) != VK_SUCCESS) {
            continue;
        }
        std::vector<VkExtensionProperties> extensions(extensionCount);
        if (enumerateDeviceExtensions(device, nullptr, &extensionCount,
                                      extensions.data()) != VK_SUCCESS) {
            continue;
        }
        const auto deviceHas = [&](const char* name) {
            return std::any_of(extensions.begin(), extensions.end(),
                               [name](const VkExtensionProperties& extension) {
                                   return std::strcmp(extension.extensionName, name) == 0;
                               });
        };
        const auto featureEnabled = [&](auto* maintenance) {
            VkPhysicalDeviceFeatures2 features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, maintenance, {}};
            getFeatures2(device, &features);
            return maintenance->swapchainMaintenance1 == VK_TRUE;
        };
#ifdef VK_KHR_swapchain_maintenance1
        if (khrSurface &&
            deviceHas(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
            VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenance{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
                nullptr, VK_FALSE};
            if (featureEnabled(&maintenance)) {
                return {RhiBackend::Vulkan,
                        RhiAvailability::ReadyForInitialization, {}};
            }
        }
#endif
#ifdef VK_EXT_swapchain_maintenance1
        if (extSurface &&
            deviceHas(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
            VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT maintenance{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
                nullptr, VK_FALSE};
            if (featureEnabled(&maintenance)) {
                return {RhiBackend::Vulkan,
                        RhiAvailability::ReadyForInitialization, {}};
            }
        }
#endif
    }
    return unavailable(
        "No Vulkan device exposes swapchain-maintenance1 presentation fences "
        "(VK_KHR/EXT_swapchain_maintenance1 with the swapchainMaintenance1 "
        "feature); native Vulkan is unavailable on this driver");
#endif
}

RhiBackendAvailability ProbeVulkanRuntimeAvailability() {
    try {
        return ProbeVulkanWsiAvailability(GetVulkanLoader());
    } catch (const std::exception& error) {
        return {RhiBackend::Vulkan, RhiAvailability::RuntimeUnavailable,
                error.what()};
    }
}

}  // namespace My
