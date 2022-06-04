#include "VulkanRHI.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <vector>

#include "geommath.hpp"
#include "portable.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

static std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static bool hasPortabilityEnumeration = false;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

std::vector<const char*> getRequiredExtensions() {
    std::vector<const char*> extensions;

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

using namespace My;

struct UniformBufferObject {
    Matrix4X4f model;
    Matrix4X4f view;
    Matrix4X4f proj;
};

VulkanRHI::VulkanRHI() {
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::cout << extensionCount << " extensions supported\n";

    std::vector<VkExtensionProperties> extensions(extensionCount);

    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "available extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << '\t' << extension.extensionName << std::endl;
        if (strncmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE) == 0) {
            hasPortabilityEnumeration = true;
        }
    }
}

VulkanRHI::~VulkanRHI() {
    vkDeviceWaitIdle(m_vkDevice);

    cleanupSwapChain(); 

    vkDestroySampler(m_vkDevice, m_vkTextureSampler, nullptr);

    vkDestroyImageView(m_vkDevice, m_vkTextureImageView, nullptr);

    vkDestroyImage(m_vkDevice, m_vkTextureImage, nullptr);
    vkFreeMemory(m_vkDevice, m_vkTextureImageMemory, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(m_vkDevice, m_vkUniformBuffers[i], nullptr);
        vkFreeMemory(m_vkDevice, m_vkUniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(m_vkDevice, m_vkDescriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayout, nullptr);

    vkDestroyBuffer(m_vkDevice, m_vkIndexBuffer, nullptr);
    vkFreeMemory(m_vkDevice, m_vkIndexBufferMemory, nullptr);

    vkDestroyBuffer(m_vkDevice, m_vkVertexBuffer, nullptr);
    vkFreeMemory(m_vkDevice, m_vkVertexBufferMemory, nullptr);

    vkDestroyShaderModule(m_vkDevice, m_vkFragShaderModule, nullptr);
    vkDestroyShaderModule(m_vkDevice, m_vkVertShaderModule, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(m_vkDevice, m_vkImageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(m_vkDevice, m_vkRenderFinishedSemaphores[i], nullptr);
        vkDestroyFence(m_vkDevice, m_vkInFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(m_vkDevice, m_vkCommandPool, nullptr);
    vkDestroyCommandPool(m_vkDevice, m_vkCommandPoolTransfer, nullptr);

    vkDestroyDevice(m_vkDevice, nullptr);     // 销毁逻辑设备

    vkDestroySurfaceKHR(m_vkInstance, m_vkSurface, nullptr);

    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugMessenger, nullptr);
    }

    vkDestroyInstance(m_vkInstance, nullptr); // 会在销毁instance的同时销毁物理设备对象
}

void VulkanRHI::createInstance (std::vector<const char*> extensions) {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    if (hasPortabilityEnumeration) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    std::cout << "create instance with extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << '\t' << extension << std::endl;
    }

    VkApplicationInfo appInfo {};
    appInfo.sType               = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName    = "Hello, Vulkan!";
    appInfo.applicationVersion  = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName         = "GEFS";
    appInfo.engineVersion       = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion          = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    createInfo.enabledExtensionCount    = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames  = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo {};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &m_vkInstance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void VulkanRHI::setupDebugMessenger() {
    if (enableValidationLayers){
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(m_vkInstance, &createInfo, nullptr, &m_vkDebugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
}

void VulkanRHI::createSurface(const std::function<void(const VkInstance&, VkSurfaceKHR&)>& createSurfaceKHR) {
    createSurfaceKHR(m_vkInstance, m_vkSurface);
}

// 一些辅助结构体和辅助方法
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> transferFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
    }
};

static QueueFamilyIndices findQueueFamilies (const VkPhysicalDevice& device, const VkSurfaceKHR surface) {
    QueueFamilyIndices _indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (!_indices.graphicsFamily.has_value() && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            _indices.graphicsFamily = i;
        } else if (!_indices.transferFamily.has_value() && queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) {
            _indices.transferFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (!_indices.presentFamily.has_value() && presentSupport) {
            _indices.presentFamily = i;
        }

        if (_indices.isComplete()) break;

        i++;
    }

    if (!_indices.transferFamily.has_value()) {
        _indices.transferFamily = _indices.graphicsFamily;
    }

    return _indices;
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

static SwapChainSupportDetails querySwapChainSupport (const VkPhysicalDevice& device, const VkSurfaceKHR& surface) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount > 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount > 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
};

void VulkanRHI::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, devices.data());

    // 设备能力检测用辅助函数
    auto checkDeviceExtensionSupport = [](VkPhysicalDevice device) -> bool {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    };

    // 设备能力检查
    auto isDeviceSuitable = [&](const VkPhysicalDevice& device, const VkSurfaceKHR& surface) -> bool {
        auto _indices = findQueueFamilies(device, surface);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return _indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    };

    for (const auto& device : devices) {
        if (isDeviceSuitable(device, m_vkSurface)) {
            m_vkPhysicalDevice = device;
            m_vkMsaaSamples = getMaxUsableSampleCount();
            break;
        }
    }

    if (m_vkPhysicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("faild to find a suitable GPU!");
    }
}

void VulkanRHI::createLogicalDevice () {
    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies;
    if (indices.graphicsFamily.has_value()) {
        uniqueQueueFamilies.emplace(indices.graphicsFamily.value());
    }
    if (indices.presentFamily.has_value()) {
        uniqueQueueFamilies.emplace(indices.presentFamily.value());
    }
    if (indices.transferFamily.has_value()) {
        uniqueQueueFamilies.emplace(indices.transferFamily.value());
    }

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType               = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex    = queueFamily;
        queueCreateInfo.queueCount          = 1;
        queueCreateInfo.pQueuePriorities    = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = m_bUseSampleShading;

    auto extensions = deviceExtensions;
    if (hasPortabilityEnumeration) {
        extensions.push_back("VK_KHR_portability_subset");
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                    = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos        = queueCreateInfos.data();
    createInfo.queueCreateInfoCount     = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures         = &deviceFeatures;
    createInfo.enabledExtensionCount    = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames   = extensions.data();
    createInfo.enabledLayerCount        = 0;

    if (vkCreateDevice(m_vkPhysicalDevice, &createInfo, nullptr, &m_vkDevice) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
}

static const VkSurfaceFormatKHR& chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB 
        && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

static const VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseSwapExtent(const int width, const int height, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

void VulkanRHI::createSwapChain () {
    int width, height;

    m_fQueryFramebufferSize(width, height);

    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_vkPhysicalDevice, m_vkSurface);

    VkSurfaceFormatKHR  surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR    presentMode   = chooseSwapPresentMode  (swapChainSupport.presentModes);
    VkExtent2D          extent        = chooseSwapExtent       (width, height, swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && 
        imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = m_vkSurface;
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode         = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount    = 2;
        createInfo.pQueueFamilyIndices      = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount    = 0;
        createInfo.pQueueFamilyIndices      = nullptr;
        createInfo.preTransform             = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha           = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode              = presentMode;
        createInfo.clipped                  = VK_TRUE;
        createInfo.oldSwapchain             = VK_NULL_HANDLE;
    }

    if (vkCreateSwapchainKHR(m_vkDevice, &createInfo, nullptr, &m_vkSwapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(m_vkDevice, m_vkSwapChain, &imageCount, nullptr);
    m_vkSwapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_vkDevice, m_vkSwapChain, &imageCount, m_vkSwapChainImages.data());

    m_vkSwapChainImageFormat = surfaceFormat.format;
    m_vkSwapChainExtent      = extent;
}

void VulkanRHI::createImageViews() {
    m_vkSwapChainImageViews.resize(m_vkSwapChainImages.size());

    for(size_t i = 0; i < m_vkSwapChainImages.size(); i++) {
        m_vkSwapChainImageViews[i] = createImageView(m_vkSwapChainImages[i], m_vkSwapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void VulkanRHI::getDeviceQueues() {
    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    vkGetDeviceQueue(m_vkDevice, indices.graphicsFamily.value(), 0, &m_vkGraphicsQueue);
    vkGetDeviceQueue(m_vkDevice, indices.presentFamily.value(), 0, &m_vkPresentQueue);
    vkGetDeviceQueue(m_vkDevice, indices.transferFamily.value(), 0, &m_vkTransferQueue);
}

void VulkanRHI::createRenderPass() {
    VkAttachmentDescription colorAttachment {};
    colorAttachment.format  = m_vkSwapChainImageFormat;
    colorAttachment.samples = m_vkMsaaSamples;
    colorAttachment.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment {};
    depthAttachment.format   = findDepthFormat();
    depthAttachment.samples  = m_vkMsaaSamples;
    depthAttachment.loadOp   = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp  = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve {};
    colorAttachmentResolve.format  = m_vkSwapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentResolveRef {};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkSubpassDependency dependency {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};

    VkRenderPassCreateInfo renderPassInfo {};
    renderPassInfo.sType            = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount  = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments     = attachments.data();
    renderPassInfo.subpassCount     = 1;
    renderPassInfo.pSubpasses       = &subpass;
    renderPassInfo.dependencyCount  = 1;
    renderPassInfo.pDependencies    = &dependency;

    if (vkCreateRenderPass(m_vkDevice, &renderPassInfo, nullptr, &m_vkRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

static VkShaderModule createShaderModule(const VkDevice& device, const Buffer& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType        = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize     = code.GetDataSize();
    createInfo.pCode        = reinterpret_cast<const uint32_t*>(code.GetData());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

void VulkanRHI::setShaders(const Buffer& vertShaderCode, const Buffer& fragShaderCode) {
    m_vkVertShaderModule = createShaderModule(m_vkDevice, vertShaderCode);
    m_vkFragShaderModule = createShaderModule(m_vkDevice, fragShaderCode);
}

void VulkanRHI::createGraphicsPipeline() {
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage   = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module  = m_vkVertShaderModule;
    vertShaderStageInfo.pName   = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage   = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module  = m_vkFragShaderModule;
    fragShaderStageInfo.pName   = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // 顶点输入格式
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount   = 1;
    vertexInputInfo.pVertexBindingDescriptions      = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions    = attributeDescriptions.data();

    // IA
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType     = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 视口和裁剪
    VkViewport viewport {};
    viewport.x      =   0.0f;
    viewport.y      =   0.0f;
    viewport.width  =   (float) m_vkSwapChainExtent.width;
    viewport.height =   (float) m_vkSwapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor {};
    scissor.offset  =   {0, 0};
    scissor.extent  =   m_vkSwapChainExtent;

    VkPipelineViewportStateCreateInfo   viewportState {};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    // 光栅化器
    VkPipelineRasterizationStateCreateInfo  rasterizer {};
    rasterizer.sType            = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode      = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth        = 1.0f;
    rasterizer.cullMode         = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable  = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp   = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // 多采样（Multisampling）
    VkPipelineMultisampleStateCreateInfo multisampling {};
    multisampling.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples  = m_vkMsaaSamples;
    multisampling.sampleShadingEnable   = m_bUseSampleShading;
    multisampling.minSampleShading      = 0.2f;
    multisampling.pSampleMask           = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable      = VK_FALSE;

    // 深度以及蒙板测试
    VkPipelineDepthStencilStateCreateInfo depthStencil {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};
    depthStencil.back = {};
    
    // 色彩混合
    VkPipelineColorBlendAttachmentState colorBlendAttachment {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable    = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.colorBlendOp   = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp   = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending {};
    colorBlending.sType                 = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable         = VK_FALSE;
    colorBlending.logicOp               = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount       = 1;
    colorBlending.pAttachments          = &colorBlendAttachment;
    colorBlending.blendConstants[0]     = 0.0f;
    colorBlending.blendConstants[1]     = 0.0f;
    colorBlending.blendConstants[2]     = 0.0f;
    colorBlending.blendConstants[3]     = 0.0f;

    // 动态（间接）管线状态
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo    dynamicState {};
    dynamicState.sType                  = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount      = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates         = dynamicStates.data();
    
    // 常量布局
    VkPipelineLayoutCreateInfo  pipelineLayoutInfo {};
    pipelineLayoutInfo.sType            = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount   = 1;
    pipelineLayoutInfo.pSetLayouts      = &m_vkDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(m_vkDevice, &pipelineLayoutInfo, nullptr, &m_vkPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("faild to create pipeline layout!");
    }

    // 创建图形渲染管道
    VkGraphicsPipelineCreateInfo pipelineInfo {};
    pipelineInfo.sType      = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages    = shaderStages;
    pipelineInfo.pVertexInputState      = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState    = &inputAssembly;
    pipelineInfo.pViewportState         = &viewportState;
    pipelineInfo.pRasterizationState    = &rasterizer;
    pipelineInfo.pMultisampleState      = &multisampling;
    pipelineInfo.pDepthStencilState     = &depthStencil;
    pipelineInfo.pColorBlendState       = &colorBlending;
    pipelineInfo.pDynamicState          = nullptr;
    pipelineInfo.layout                 = m_vkPipelineLayout;
    pipelineInfo.renderPass             = m_vkRenderPass;
    pipelineInfo.subpass                = 0;
    pipelineInfo.basePipelineHandle     = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex      = -1;

    if (vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_vkGraphicPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
}

void VulkanRHI::createFramebuffers() {
    m_vkSwapChainFramebuffers.resize(m_vkSwapChainImageViews.size());

    for (size_t i = 0; i < m_vkSwapChainImageViews.size(); i++) {
        std::array<VkImageView, 3> attachments = {
            m_vkColorImageView,
            m_vkDepthImageView,
            m_vkSwapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo {};
        framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass      = m_vkRenderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments    = attachments.data();
        framebufferInfo.width           = m_vkSwapChainExtent.width;
        framebufferInfo.height          = m_vkSwapChainExtent.height;
        framebufferInfo.layers          = 1;

        if (vkCreateFramebuffer(m_vkDevice, &framebufferInfo, nullptr, &m_vkSwapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanRHI::createCommandPool() {
    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = indices.graphicsFamily.value();

    if (vkCreateCommandPool(m_vkDevice, &poolInfo, nullptr, &m_vkCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    // create a secondary command pool for command buffers submits to transfer queue family
    poolInfo.queueFamilyIndex =indices.transferFamily.value();

    if (vkCreateCommandPool(m_vkDevice, &poolInfo, nullptr, &m_vkCommandPoolTransfer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create transfer command pool!");
    }

}

void VulkanRHI::createCommandBuffers() {
    m_vkCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_vkCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) m_vkCommandBuffers.size();

    if (vkAllocateCommandBuffers(m_vkDevice, &allocInfo, m_vkCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void VulkanRHI::recordCommandBuffer(VkCommandBuffer& commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_vkRenderPass;
    renderPassInfo.framebuffer = m_vkSwapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_vkSwapChainExtent;

    std::array<VkClearValue, 2> clearValues {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkGraphicPipeline);

    VkBuffer vertexBuffers[] = {m_vkVertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(m_vkCommandBuffers[m_nCurrentFrame], 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(m_vkCommandBuffers[m_nCurrentFrame], m_vkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(m_vkCommandBuffers[m_nCurrentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkPipelineLayout, 0, 1, &m_vkDescriptorSets[m_nCurrentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_Indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void VulkanRHI::createSyncObjects() {
    m_vkImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_vkRenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_vkInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(m_vkDevice, &semaphoreInfo, nullptr, &m_vkImageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(m_vkDevice, &semaphoreInfo, nullptr, &m_vkRenderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(m_vkDevice, &fenceInfo, nullptr, &m_vkInFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
        }
    }
}

void VulkanRHI::drawFrame() {
    vkWaitForFences(m_vkDevice, 1, &m_vkInFlightFences[m_nCurrentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(m_vkDevice, 1, &m_vkInFlightFences[m_nCurrentFrame]);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(m_vkDevice, m_vkSwapChain, UINT64_MAX, m_vkImageAvailableSemaphores[m_nCurrentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vkResetCommandBuffer(m_vkCommandBuffers[m_nCurrentFrame], 0);
    recordCommandBuffer(m_vkCommandBuffers[m_nCurrentFrame], imageIndex);
    
    // 更新常量
    updateUniformBufer(m_nCurrentFrame);

    // 提交 Command Buffer
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {m_vkImageAvailableSemaphores[m_nCurrentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_vkCommandBuffers[m_nCurrentFrame];

    VkSemaphore signalSemaphores[] = {m_vkRenderFinishedSemaphores[m_nCurrentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(m_vkGraphicsQueue, 1, &submitInfo, m_vkInFlightFences[m_nCurrentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {m_vkSwapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(m_vkPresentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    m_nCurrentFrame = (m_nCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRHI::cleanupSwapChain() {
    vkDestroyImageView(m_vkDevice, m_vkColorImageView, nullptr);
    vkDestroyImage(m_vkDevice, m_vkColorImage, nullptr);
    vkFreeMemory(m_vkDevice, m_vkColorImageMemory, nullptr);

    vkDestroyImageView(m_vkDevice, m_vkDepthImageView, nullptr);
    vkDestroyImage(m_vkDevice, m_vkDepthImage, nullptr);
    vkFreeMemory(m_vkDevice, m_vkDepthImageMemory, nullptr);

    for (auto& framebuffer : m_vkSwapChainFramebuffers) {
        vkDestroyFramebuffer(m_vkDevice, framebuffer, nullptr);
    }

    vkDestroyPipeline(m_vkDevice, m_vkGraphicPipeline, nullptr);

    vkDestroyPipelineLayout(m_vkDevice, m_vkPipelineLayout, nullptr);

    vkDestroyRenderPass(m_vkDevice, m_vkRenderPass, nullptr);

    for (auto& imageView : m_vkSwapChainImageViews) {
        vkDestroyImageView(m_vkDevice, imageView, nullptr);
    }

    vkDestroySwapchainKHR(m_vkDevice, m_vkSwapChain, nullptr);
}

void VulkanRHI::recreateSwapChain() {
    vkDeviceWaitIdle(m_vkDevice);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createColorResources();
    createDepthResources();
    createFramebuffers();
}

uint32_t VulkanRHI::findMemoryType (uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_vkPhysicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanRHI::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = size;
    bufferInfo.usage = usage;
    bufferInfo.flags = 0;

    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    if (indices.graphicsFamily.value() != indices.transferFamily.value()) {
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.transferFamily.value()};

        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = 2;
        bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value()};

        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.queueFamilyIndexCount = 1;
        bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
        bufferInfo.flags = 0;
    }

    if (vkCreateBuffer(m_vkDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_vkDevice, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_vkDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory");
    }

    vkBindBufferMemory(m_vkDevice, buffer, bufferMemory, 0);
}

void VulkanRHI::createVertexBuffer() {
    VkDeviceSize bufferSize = sizeof(m_Vertices[0]) * m_Vertices.size();

    // 创建中间缓冲区
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    // 上传数据
    void* data;
    vkMapMemory(m_vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, m_Vertices.data(), (size_t) bufferSize);
    vkUnmapMemory(m_vkDevice, stagingBufferMemory);

    // 创建设备专有缓冲区
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vkVertexBuffer, m_vkVertexBufferMemory);

    copyBuffer(stagingBuffer, m_vkVertexBuffer, bufferSize);

    vkDestroyBuffer(m_vkDevice, stagingBuffer, nullptr);
    vkFreeMemory(m_vkDevice, stagingBufferMemory, nullptr);
} 

void VulkanRHI::createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(m_Indices[0]) * m_Indices.size();

    // 创建中间缓冲区
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    // 上传数据
    void* data;
    vkMapMemory(m_vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, m_Indices.data(), (size_t) bufferSize);
    vkUnmapMemory(m_vkDevice, stagingBufferMemory);

    // 创建设备专有缓冲区
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vkIndexBuffer, m_vkIndexBufferMemory);

    copyBuffer(stagingBuffer, m_vkIndexBuffer, bufferSize);

    vkDestroyBuffer(m_vkDevice, stagingBuffer, nullptr);
    vkFreeMemory(m_vkDevice, stagingBufferMemory, nullptr);
} 

void VulkanRHI::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    auto commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    endSingleTimeCommands(commandBuffer);
}

void VulkanRHI::copyBufferToImage(VkBuffer srcBuffer, VkImage image, uint32_t width, uint32_t height) {
    auto commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(commandBuffer, srcBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

void VulkanRHI::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding samplerLayoutBinding {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_vkDevice, &layoutInfo, nullptr, &m_vkDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void VulkanRHI::createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    m_vkUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    m_vkUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_vkUniformBuffers[i], m_vkUniformBuffersMemory[i]);
    }
}

void VulkanRHI::updateUniformBufer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo {};
    BuildIdentityMatrix(ubo.model);
    MatrixRotationAxis(ubo.model, {0.0f, 0.0f, 1.0f}, time * PI / 8.0f);
    BuildViewRHMatrix(ubo.view, {2.0f, 2.0f, 2.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f});
    BuildPerspectiveFovRHMatrix(ubo.proj, PI / 4.0f, m_vkSwapChainExtent.width / m_vkSwapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1.0f;

    // 上传数据
    void* data;
    vkMapMemory(m_vkDevice, m_vkUniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(m_vkDevice, m_vkUniformBuffersMemory[currentImage]);
}

void VulkanRHI::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSize {};
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSize[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    poolInfo.pPoolSizes = poolSize.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolInfo.flags = 0;

    if (vkCreateDescriptorPool(m_vkDevice, &poolInfo, nullptr, &m_vkDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void VulkanRHI::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_vkDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_vkDescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    m_vkDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(m_vkDevice, &allocInfo, m_vkDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo {};
        bufferInfo.buffer = m_vkUniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_vkTextureImageView;
        imageInfo.sampler = m_vkTextureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites {};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = m_vkDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;
        descriptorWrites[0].pImageInfo = nullptr;
        descriptorWrites[0].pTexelBufferView = nullptr;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = m_vkDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = nullptr;
        descriptorWrites[1].pImageInfo = &imageInfo;
        descriptorWrites[1].pTexelBufferView = nullptr;

        vkUpdateDescriptorSets(m_vkDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

}

void VulkanRHI::getTextureFormat(const Image& img, VkFormat& internal_format) {
    if (img.compressed) {
        switch (img.compress_format) {
            case "BC1_"_u32:
                internal_format = VK_FORMAT_BC1_RGB_SRGB_BLOCK;
                break;
            case "BC1A"_u32:
                internal_format = VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
                break;
            case "BC2_"_u32:
                internal_format = VK_FORMAT_BC2_SRGB_BLOCK;
                break;
            case "BC3_"_u32:
                internal_format = VK_FORMAT_BC3_SRGB_BLOCK;
                break;
            case "BC4S"_u32:
                internal_format = VK_FORMAT_BC4_SNORM_BLOCK;
                break;
            case "BC4U"_u32:
                internal_format = VK_FORMAT_BC4_UNORM_BLOCK;
                break;
            case "BC5S"_u32:
                internal_format = VK_FORMAT_BC5_SNORM_BLOCK;
                break;
            case "BC5U"_u32:
                internal_format = VK_FORMAT_BC5_UNORM_BLOCK;
                break;
            case "BC6S"_u32:
                internal_format = VK_FORMAT_BC6H_SFLOAT_BLOCK;
                break;
            case "BC6U"_u32:
                internal_format = VK_FORMAT_BC6H_UFLOAT_BLOCK;
                break;
            case "BC7S"_u32:
                internal_format = VK_FORMAT_BC7_SRGB_BLOCK;
                break;
            case "BC7U"_u32:
                internal_format = VK_FORMAT_BC7_UNORM_BLOCK;
                break;
            default:
                assert(0);
        }
    } else {
        if (img.bitcount == 8) {
            internal_format = VK_FORMAT_R8_UNORM;
        } else if (img.bitcount == 16) {
            internal_format = VK_FORMAT_R16_UNORM;
        } else if (img.bitcount == 24) {
            internal_format = VK_FORMAT_R8G8B8_SRGB;
        } else if (img.bitcount == 32) {
            internal_format = VK_FORMAT_R8G8B8A8_UNORM;
        } else if (img.bitcount == 64) {
            if (img.is_float) {
                internal_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            } else {
                internal_format = VK_FORMAT_R16G16B16A16_UNORM;
            }
        } else if (img.bitcount == 128) {
            if (img.is_float) {
                internal_format = VK_FORMAT_R32G32B32A32_SFLOAT;
            } else {
                internal_format = VK_FORMAT_R32G32B32A32_UINT;
            }
        } else {
            assert(0);
        }
    }
}

void VulkanRHI::createImage(Image& image, VkImageTiling tiling, VkImageUsageFlags usage, VkSharingMode sharing_mode, VkMemoryPropertyFlags properties, VkImage& vk_image, VkDeviceMemory& vk_image_memory, uint32_t queueFamilyIndexCount, uint32_t* queueFamilyIndices) {
    if (!image.data) {
        throw std::runtime_error("faild to load texture image!");
    }

    VkFormat format;
    getTextureFormat(image, format);
    createImage(image.Width, image.Height, VK_SAMPLE_COUNT_1_BIT, format, tiling, usage, sharing_mode, properties, vk_image, vk_image_memory, queueFamilyIndexCount, queueFamilyIndices);
}

void VulkanRHI::createImage(uint32_t width, uint32_t height, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkSharingMode sharing_mode, VkMemoryPropertyFlags properties, VkImage& vk_image, VkDeviceMemory& vk_image_memory, uint32_t queueFamilyIndexCount, uint32_t* queueFamilyIndices) {
    VkImageCreateInfo imageInfo {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;

    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = sharing_mode;
    if (queueFamilyIndexCount > 0) {
        imageInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        imageInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    imageInfo.samples = numSamples;
    imageInfo.flags = 0;

    if (vkCreateImage(m_vkDevice, &imageInfo, nullptr, &vk_image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_vkDevice, vk_image, &memRequirements);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_vkDevice, &allocInfo, nullptr, &vk_image_memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(m_vkDevice, vk_image, vk_image_memory, 0);
}

void VulkanRHI::createTextureImage(Image& image) {
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(image.data_size);

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    // 上传数据
    void* data;
    vkMapMemory(m_vkDevice, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, image.data, image.data_size);
    vkUnmapMemory(m_vkDevice, stagingBufferMemory);

    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);
    bool useSeperateTransferFamily = (indices.transferFamily.value() != indices.graphicsFamily.value());

    if (useSeperateTransferFamily) {
        std::array<uint32_t, 2> queueFamilyIndices = {indices.transferFamily.value(), indices.graphicsFamily.value()}; 
        createImage(image, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_CONCURRENT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vkTextureImage, m_vkTextureImageMemory, static_cast<uint32_t>(queueFamilyIndices.size()), queueFamilyIndices.data());
    } else {
        std::array<uint32_t, 1> queueFamilyIndices = {indices.graphicsFamily.value()}; 
        createImage(image, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vkTextureImage, m_vkTextureImageMemory, static_cast<uint32_t>(queueFamilyIndices.size()), queueFamilyIndices.data());
    }

    VkFormat format;
    getTextureFormat(image, format);

    transitionImageLayout(m_vkTextureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    copyBufferToImage(stagingBuffer, m_vkTextureImage, image.Width, image.Height);

    transitionImageLayout(m_vkTextureImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(m_vkDevice, stagingBuffer, nullptr);
    vkFreeMemory(m_vkDevice, stagingBufferMemory, nullptr);
}

VkImageView VulkanRHI::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(m_vkDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

void VulkanRHI::createTextureImageView(Image& image) {
    VkFormat format;
    getTextureFormat(image, format);
    m_vkTextureImageView = createImageView(m_vkTextureImage, format, VK_IMAGE_ASPECT_COLOR_BIT);
}

void VulkanRHI::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;

    VkPhysicalDeviceProperties properties {};
    vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(m_vkDevice, &samplerInfo, nullptr, &m_vkTextureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

VkCommandBuffer VulkanRHI::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_vkCommandPoolTransfer;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_vkDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanRHI::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(m_vkTransferQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_vkTransferQueue);
    
    vkFreeCommandBuffers(m_vkDevice, m_vkCommandPoolTransfer, 1, &commandBuffer);
}

void VulkanRHI::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, 
    uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex) {

    auto commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
    barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer, 
                        sourceStage, destinationStage, 
                        0,
                        0, nullptr, 
                        0, nullptr, 
                        1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

VkFormat VulkanRHI::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

VkFormat VulkanRHI::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

static bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void VulkanRHI::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();

    createImage(m_vkSwapChainExtent.width, m_vkSwapChainExtent.height, m_vkMsaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vkDepthImage, m_vkDepthImageMemory);
    m_vkDepthImageView = createImageView(m_vkDepthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}

VkSampleCountFlagBits VulkanRHI::getMaxUsableSampleCount() {
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

void VulkanRHI::createColorResources() {
    VkFormat colorFormat = m_vkSwapChainImageFormat;

    createImage(m_vkSwapChainExtent.width, m_vkSwapChainExtent.height, m_vkMsaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vkColorImage, m_vkColorImageMemory);
    m_vkColorImageView = createImageView(m_vkColorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
}

void VulkanRHI::setModel(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
    m_Vertices = vertices;
    m_Indices = indices;
}