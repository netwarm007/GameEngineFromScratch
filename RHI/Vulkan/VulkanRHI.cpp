#include "VulkanRHI.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <vector>

#include "geommath.hpp"
#include "portable.hpp"

const uint32_t MAX_FRAMES_IN_FLIGHT = My::GfxConfiguration::kMaxInFlightFrameCount;

static std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

static void populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugMessageFunc(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                 VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                 void* pUserData) {
    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && pUserData) {
        ++*static_cast<std::atomic_uint*>(pUserData);
    }
    std::cerr << "Vulkan validation: " << pCallbackData->pMessage << '\n';

    return false;
}

std::vector<const char*> getRequiredExtensions() {
    std::vector<const char*> extensions;

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool checkValidationLayerSupport() {
    const auto layers = vk::enumerateInstanceLayerProperties();
    return std::any_of(layers.begin(), layers.end(), [](const auto& layer) {
        return std::strcmp(layer.layerName, validationLayers.front()) == 0;
    });
}

using namespace My;

namespace {
struct ScopedBuffer {
    explicit ScopedBuffer(vk::Device value) : device(value) {}
    ScopedBuffer(const ScopedBuffer&) = delete;
    ScopedBuffer& operator=(const ScopedBuffer&) = delete;
    ~ScopedBuffer() {
        device.destroyBuffer(buffer);
        device.freeMemory(memory);
    }
    void Release() {
        buffer = nullptr;
        memory = nullptr;
    }
    vk::Device device;
    vk::Buffer buffer;
    vk::DeviceMemory memory;
};
}  // namespace

VulkanRHI::VulkanRHI() = default;

VulkanRHI::~VulkanRHI() { destroyAll(); }

void VulkanRHI::Initialize(const std::vector<const char*>& extensions,
                           const CreateSurfaceCBFunc& surface) {
    if (m_vkInstance) {
        throw std::logic_error("Vulkan RHI is already initialized");
    }
    try {
        createInstance(extensions);
        setupDebugMessenger();
        createSurface(surface);
        pickPhysicalDevice();
        createLogicalDevice();
        getDeviceQueues();
        createCommandPool();
        createSwapChain();
        createSyncObjects();
        createCommandBuffers();
    } catch (const std::exception&) {
        destroyAll();
        throw;
    }
}

void VulkanRHI::WaitGPU() {
    if (m_vkDevice) m_vkDevice.waitIdle();
}

void VulkanRHI::destroyAll() {
    if (m_vkDevice) {
        try {
            WaitGPU();
        } catch (const vk::SystemError& error) {
            std::cerr << "Vulkan shutdown wait failed: " << error.what() << '\n';
        }

        cleanupSwapChain();

        m_vkDevice.destroyDescriptorPool(m_vkDescriptorPool);
        m_vkDescriptorPool = nullptr;
        m_vkDescriptorSets.clear();

        for (auto semaphore : m_vkImageAvailableSemaphores) {
            m_vkDevice.destroySemaphore(semaphore);
        }

        for (auto fence : m_vkInFlightFences) {
            m_vkDevice.destroyFence(fence);
        }

        m_vkImageAvailableSemaphores.clear();
        m_vkRenderFinishedSemaphores.clear();
        m_vkInFlightFences.clear();
        m_vkCommandBuffers.clear();
        m_vkDevice.destroyCommandPool(m_vkCommandPool);
        m_vkDevice.destroyCommandPool(m_vkCommandPoolTransfer);
        m_vkCommandPool = nullptr;
        m_vkCommandPoolTransfer = nullptr;

        m_vkDevice.destroy();
        m_vkDevice = nullptr;
    }

    if (m_vkInstance) {
        m_vkInstance.destroySurfaceKHR(m_vkSurface);
        m_vkSurface = nullptr;
        if (m_vkDebugMessenger && m_destroyDebugMessenger) {
            m_destroyDebugMessenger(m_vkInstance, m_vkDebugMessenger, nullptr);
        }
        m_vkDebugMessenger = nullptr;
        m_destroyDebugMessenger = nullptr;
        m_vkInstance.destroy();
        m_vkInstance = nullptr;
    }
    m_vkPhysicalDevice = nullptr;
    m_vkGraphicsQueue = nullptr;
    m_vkComputeQueue = nullptr;
    m_vkPresentQueue = nullptr;
    m_vkTransferQueue = nullptr;
    m_nCurrentFrame = 0;
    m_currentImage = 0;
    m_frameActive = false;
    m_recreateSwapChain = false;
}

void VulkanRHI::createInstance(std::vector<const char*> extensions) {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error(
            "validation layers requested, but not available!");
    }

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    const auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    for (const char* extension : extensions) {
        if (std::none_of(availableExtensions.begin(), availableExtensions.end(),
                         [extension](const auto& available) {
                             return std::strcmp(available.extensionName,
                                                extension) == 0;
                         })) {
            throw std::runtime_error(std::string("Required Vulkan instance extension unavailable: ") + extension);
        }
    }

    std::cout << "create instance with extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << '\t' << extension << std::endl;
    }

    // 初始化 vk::ApplicationInfo 结构体
    vk::ApplicationInfo applicationInfo(
        "Hello, Vulkan!", VK_MAKE_VERSION(1, 0, 0), "GEFS",
        VK_MAKE_VERSION(0, 1, 0), VK_API_VERSION_1_1);

    vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo, {}, extensions);
    if (enableValidationLayers) instanceCreateInfo.setPEnabledLayerNames(validationLayers);
    const vk::DebugUtilsMessengerCreateInfoEXT debugInfo(
        {}, vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        debugMessageFunc, &m_validationErrors);
    const VkValidationFeatureEnableEXT sync =
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    const VkValidationFeaturesEXT validation{
        VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, &debugInfo, 1, &sync, 0, nullptr};
    if (enableValidationLayers) instanceCreateInfo.pNext = &validation;
#ifdef VK_KHR_portability_enumeration
    if (std::any_of(extensions.begin(), extensions.end(), [](const char* extension) {
            return std::strcmp(extension, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        })) {
        instanceCreateInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
#endif

    m_vkInstance = vk::createInstance(instanceCreateInfo);
}

void VulkanRHI::setupDebugMessenger() {
    if (enableValidationLayers) {
        const auto createDebugMessenger =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                m_vkInstance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
        m_destroyDebugMessenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                m_vkInstance.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
        if (!createDebugMessenger || !m_destroyDebugMessenger) {
            throw std::runtime_error("Vulkan debug utils entry points unavailable");
        }

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        const vk::DebugUtilsMessengerCreateInfoEXT info(
            {}, severityFlags, messageTypeFlags, &debugMessageFunc, &m_validationErrors);
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        const auto result = createDebugMessenger(
            m_vkInstance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&info),
            nullptr, &messenger);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan debug messenger: " +
                                     std::to_string(result));
        }
        m_vkDebugMessenger = messenger;
    }
}

void VulkanRHI::createSurface(
    const CreateSurfaceCBFunc&
        createSurfaceKHR) {
    if (!m_vkInstance || !createSurfaceKHR) {
        throw std::logic_error("Vulkan surface requires an instance and callback");
    }
    createSurfaceKHR(m_vkInstance, m_vkSurface);
    if (!m_vkSurface) throw std::runtime_error("Vulkan surface callback returned no surface");
}

// 一些辅助结构体和辅助方法
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> transferFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() &&
               transferFamily.has_value();
    }
};

static QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device,
                                            const VkSurfaceKHR surface) {
    QueueFamilyIndices _indices;
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        device.getQueueFamilyProperties();

    int i = 0;
    for (const auto& queueFamilyProperty : queueFamilyProperties) {
        const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
        if (!_indices.graphicsFamily.has_value() &&
            (queueFamilyProperty.queueFlags & required) == required) {
            _indices.graphicsFamily = i;
            _indices.transferFamily = i;
        }

        VkBool32 presentSupport = device.getSurfaceSupportKHR(i, surface);
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
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

static SwapChainSupportDetails querySwapChainSupport(
    const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface) {
    SwapChainSupportDetails details;

    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);

    details.formats = device.getSurfaceFormatsKHR(surface);

    details.presentModes = device.getSurfacePresentModesKHR(surface);

    return details;
};

void VulkanRHI::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices =
        m_vkInstance.enumeratePhysicalDevices();

    if (devices.size() == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    // 设备能力检测用辅助函数
    auto checkDeviceExtensionSupport = [](vk::PhysicalDevice device) -> bool {
        std::vector<vk::ExtensionProperties> availableExtensions =
            device.enumerateDeviceExtensionProperties();

        std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                                 deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    };

    // 设备能力检查
    auto isDeviceSuitable = [&](const vk::PhysicalDevice& device,
                                const vk::SurfaceKHR& surface) -> bool {
        if (device.getProperties().apiVersion < VK_API_VERSION_1_1) return false;
        auto _indices = findQueueFamilies(device, surface);

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        if (!_indices.isComplete() || !extensionsSupported) return false;

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport =
                querySwapChainSupport(device, surface);
            swapChainAdequate = !swapChainSupport.formats.empty() &&
                                !swapChainSupport.presentModes.empty();
        }

        vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

        return _indices.isComplete() && extensionsSupported &&
               swapChainAdequate && supportedFeatures.imageCubeArray;
    };

    for (const auto& device : devices) {
        std::cout << "Checking Vulkan device: " << device.getProperties().deviceName << '\n';
        if (isDeviceSuitable(device, m_vkSurface)) {
            m_vkPhysicalDevice = device;
            m_vkMsaaSamples = getMaxUsableSampleCount();
            if (m_fGetGfxConfigCB &&
                static_cast<uint32_t>(m_vkMsaaSamples) != m_fGetGfxConfigCB().msaaSamples) {
                std::cerr << "Requested Vulkan MSAA count unsupported; using "
                          << static_cast<uint32_t>(m_vkMsaaSamples) << " samples\n";
            }
            break;
        }
    }

    if (!m_vkPhysicalDevice) {
        throw std::runtime_error("faild to find a suitable GPU!");
    }
}

void VulkanRHI::createLogicalDevice() {
    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
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
        vk::DeviceQueueCreateInfo queueCreateInfo(
            vk::DeviceQueueCreateFlags(), queueFamily, 1, &queuePriority);
        queueCreateInfos.push_back(queueCreateInfo);
    }

    const auto supported = m_vkPhysicalDevice.getFeatures();
    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceFeatures.samplerAnisotropy = supported.samplerAnisotropy;
    deviceFeatures.sampleRateShading = supported.sampleRateShading;
    deviceFeatures.imageCubeArray = VK_TRUE;
    deviceFeatures.geometryShader = supported.geometryShader;
    deviceFeatures.textureCompressionBC = supported.textureCompressionBC;
    deviceFeatures.textureCompressionETC2 = supported.textureCompressionETC2;
    deviceFeatures.textureCompressionASTC_LDR = supported.textureCompressionASTC_LDR;
    m_bUseSampleShading = deviceFeatures.sampleRateShading;

    auto extensions = deviceExtensions;

    for (const auto& extension : m_vkPhysicalDevice.enumerateDeviceExtensionProperties()) {
        if (std::strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0) {
            extensions.push_back("VK_KHR_portability_subset");
        }
    }

    vk::DeviceCreateInfo createInfo(vk::DeviceCreateFlags(), queueCreateInfos,
                                    {}, extensions, &deviceFeatures);
    m_vkDevice = m_vkPhysicalDevice.createDevice(createInfo);
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    if (availableFormats.empty()) throw std::runtime_error("Vulkan surface has no formats");
    if (availableFormats.size() == 1 && availableFormats.front().format == vk::Format::eUndefined) {
        return {vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear};
    }
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

static const vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

static vk::Extent2D chooseSwapExtent(
    const int width, const int height,
    const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        vk::Extent2D actualExtent = {static_cast<uint32_t>(width),
                                     static_cast<uint32_t>(height)};

        actualExtent.width =
            std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width);
        actualExtent.height =
            std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

void VulkanRHI::createSwapChain() {
    if (!m_fQueryFramebufferSizeCB) {
        throw std::logic_error("Vulkan swapchain requires a framebuffer size callback");
    }
    if (m_vkSwapChain) {
        throw std::logic_error("Existing Vulkan swapchain must be released before recreation");
    }
    uint32_t width = 0, height = 0;
    m_fQueryFramebufferSizeCB(width, height);
    if (width == 0 || height == 0) return;

    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(m_vkPhysicalDevice, m_vkSurface);

    vk::SurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent =
        chooseSwapExtent(width, height, swapChainSupport.capabilities);
    if (extent.width == 0 || extent.height == 0) return;

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo(
        vk::SwapchainCreateFlagsKHR(), m_vkSurface, imageCount,
        surfaceFormat.format, surfaceFormat.colorSpace, extent, 1,
        vk::ImageUsageFlagBits::eColorAttachment);

    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    std::array<uint32_t, 2> queueFamilyIndices = {
        indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        createInfo.setQueueFamilyIndices(queueFamilyIndices);
    } else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    const auto alpha = swapChainSupport.capabilities.supportedCompositeAlpha;
    for (auto candidate : {vk::CompositeAlphaFlagBitsKHR::eOpaque,
                           vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
                           vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
                           vk::CompositeAlphaFlagBitsKHR::eInherit}) {
        if (alpha & candidate) {
            createInfo.compositeAlpha = candidate;
            break;
        }
    }
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    m_vkSwapChain = m_vkDevice.createSwapchainKHR(createInfo);

    m_vkSwapChainImages = m_vkDevice.getSwapchainImagesKHR(m_vkSwapChain);

    m_vkSwapChainImageFormat = surfaceFormat.format;
    m_vkSwapChainExtent = extent;
    m_framebufferWidth = width;
    m_framebufferHeight = height;

    m_vkSwapChainImageViews.resize(m_vkSwapChainImages.size());

    for (size_t i = 0; i < m_vkSwapChainImages.size(); i++) {
        m_vkSwapChainImageViews[i] =
            createImageView(m_vkSwapChainImages[i], m_vkSwapChainImageFormat,
                            vk::ImageAspectFlagBits::eColor);
    }

    m_vkImagesInFlight.assign(m_vkSwapChainImages.size(), vk::Fence{});
    m_vkRenderFinishedSemaphores.resize(m_vkSwapChainImages.size());
    for (auto& semaphore : m_vkRenderFinishedSemaphores) {
        semaphore = m_vkDevice.createSemaphore({});
    }
    ++m_swapChainGeneration;
    m_recreateSwapChain = false;
    if (m_fCreateSwapChainCB) CreateSwapChain();
}

void VulkanRHI::getDeviceQueues() {
    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    m_vkGraphicsQueue = m_vkDevice.getQueue(indices.graphicsFamily.value(), 0);
    m_vkPresentQueue = m_vkDevice.getQueue(indices.presentFamily.value(), 0);
    m_vkTransferQueue = m_vkDevice.getQueue(indices.transferFamily.value(), 0);
    m_vkComputeQueue = m_vkGraphicsQueue;
}

vk::RenderPass VulkanRHI::createRenderPass() {
    vk::AttachmentDescription colorAttachment;
    colorAttachment.format = m_vkSwapChainImageFormat;
    colorAttachment.samples = m_vkMsaaSamples;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = m_vkMsaaSamples == vk::SampleCountFlagBits::e1
                                      ? vk::ImageLayout::ePresentSrcKHR
                                      : vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference colorAttachmentRef;
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentDescription depthAttachment;
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = m_vkMsaaSamples;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachment.finalLayout =
        vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef;
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentDescription colorAttachmentResolve;
    colorAttachmentResolve.format = m_vkSwapChainImageFormat;
    colorAttachmentResolve.samples = vk::SampleCountFlagBits::e1;
    colorAttachmentResolve.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachmentResolve.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentResolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachmentResolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachmentResolve.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachmentResolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentResolveRef;
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass;
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.setColorAttachments(colorAttachmentRef);
    subpass.setPDepthStencilAttachment(&depthAttachmentRef);
    if (m_vkMsaaSamples != vk::SampleCountFlagBits::e1) {
        subpass.setPResolveAttachments(&colorAttachmentResolveRef);
    }

    vk::SubpassDependency dependency;
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput |
        vk::PipelineStageFlagBits::eEarlyFragmentTests |
        vk::PipelineStageFlagBits::eLateFragmentTests;
    dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                               vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    dependency.dstStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput |
        vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                               vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    std::vector<vk::AttachmentDescription> attachments = {colorAttachment, depthAttachment};
    if (m_vkMsaaSamples != vk::SampleCountFlagBits::e1) {
        attachments.push_back(colorAttachmentResolve);
    }

    vk::RenderPassCreateInfo renderPassInfo(vk::RenderPassCreateFlags(),
                                            attachments, subpass, dependency);

    return m_vkDevice.createRenderPass(renderPassInfo);
}

vk::ShaderModule VulkanRHI::createShaderModule(const Buffer& code) {
    vk::ShaderModuleCreateInfo createInfo;
    createInfo.codeSize = code.GetDataSize();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.GetData());

    vk::ShaderModule shaderModule;
    shaderModule = m_vkDevice.createShaderModule(createInfo);

    return shaderModule;
}

void VulkanRHI::createGraphicsPipeline(const VulkanRHI::Shader *shaders, size_t shaderCount,
                                       const vk::VertexInputBindingDescription &bindingDescription, 
                                       const vk::VertexInputAttributeDescription *attributeDescriptions, size_t attributeCount,
                                       const vk::DescriptorSetLayout &descriptorSetLayout,
                                       const vk::RenderPass &renderPass,
                                       vk::PipelineLayout &pipelineLayout,
                                       vk::Pipeline &pipeline
                                       ) {
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;

    for (size_t i = 0; i < shaderCount; i++) {
        vk::PipelineShaderStageCreateInfo shaderStageInfo;
        shaderStageInfo.stage = shaders[i].stage;
        shaderStageInfo.module = shaders[i].module;
        shaderStageInfo.pName = shaders[i].entry_point.c_str();

        shaderStages.push_back(shaderStageInfo);
    }

    // 顶点输入格式
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeCount);
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

    // IA
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 视口和裁剪
    vk::Viewport viewport(0.0f, 0.0f, m_vkSwapChainExtent.width,
                          m_vkSwapChainExtent.height, 0.0f, 1.0f);

    vk::Rect2D scissor({0, 0}, m_vkSwapChainExtent);

    vk::PipelineViewportStateCreateInfo viewportState({}, viewport, scissor);

    // 光栅化器
    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // 多采样（Multisampling）
    vk::PipelineMultisampleStateCreateInfo multisampling;
    multisampling.rasterizationSamples = m_vkMsaaSamples;
    multisampling.sampleShadingEnable = m_bUseSampleShading;
    multisampling.minSampleShading = 0.2f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // 深度以及蒙板测试
    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;

    // 色彩混合
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.setAttachments(colorBlendAttachment);
    colorBlending.setBlendConstants({0.0f, 0.0f, 0.0f, 0.0f});

    // 动态（间接）管线状态
    std::array<vk::DynamicState, 2> states = {vk::DynamicState::eViewport,
                                              vk::DynamicState::eLineWidth};

    vk::PipelineDynamicStateCreateInfo dynamicState({}, states);

    // 常量布局
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descriptorSetLayout);

    pipelineLayout = m_vkDevice.createPipelineLayout(pipelineLayoutInfo);

    // 创建图形渲染管道
    vk::GraphicsPipelineCreateInfo pipelineInfo(
        vk::PipelineCreateFlags(), shaderStages, &vertexInputInfo,
        &inputAssembly, nullptr, &viewportState, &rasterizer, &multisampling,
        &depthStencil, &colorBlending, nullptr, pipelineLayout,
        renderPass);

    vk::Result result;
    std::tie(result, pipeline) =
        m_vkDevice.createGraphicsPipeline(nullptr, pipelineInfo);

    switch (result) {
        case vk::Result::eSuccess:
            break;
        case vk::Result::ePipelineCompileRequiredEXT:
            // something meaningfull here
            break;
        default:
            assert(false);  // should never happen
    }
}

void VulkanRHI::createFramebuffers(const vk::RenderPass &renderPass) {
    m_vkSwapChainFramebuffers.resize(m_vkSwapChainImageViews.size());

    for (size_t i = 0; i < m_vkSwapChainImageViews.size(); i++) {
        std::vector<vk::ImageView> attachments = {
            m_vkSwapChainImageViews[i], m_vkDepthImageView};
        if (m_vkMsaaSamples != vk::SampleCountFlagBits::e1) {
            attachments = {m_vkColorImageView, m_vkDepthImageView, m_vkSwapChainImageViews[i]};
        }

        vk::FramebufferCreateInfo framebufferInfo(
            {}, renderPass, attachments, m_vkSwapChainExtent.width,
            m_vkSwapChainExtent.height, 1);

        m_vkSwapChainFramebuffers[i] =
            m_vkDevice.createFramebuffer(framebufferInfo);
    }
}

void VulkanRHI::createCommandPool() {
    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);

    vk::CommandPoolCreateInfo poolInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        indices.graphicsFamily.value());

    m_vkCommandPool = m_vkDevice.createCommandPool(poolInfo);

    // create a secondary command pool for command buffers submits to transfer
    // queue family
    poolInfo.queueFamilyIndex = indices.transferFamily.value();

    m_vkCommandPoolTransfer = m_vkDevice.createCommandPool(poolInfo);
}

void VulkanRHI::createCommandBuffers() {
    m_vkCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    vk::CommandBufferAllocateInfo allocInfo(
        m_vkCommandPool, vk::CommandBufferLevel::ePrimary,
        static_cast<uint32_t>(m_vkCommandBuffers.size()));

    m_vkCommandBuffers = m_vkDevice.allocateCommandBuffers(allocInfo);
}

void VulkanRHI::recordCommandBuffer(const vk::RenderPass &renderPass,
                                    const vk::Pipeline &pipeline,
                                    const vk::PipelineLayout &pipelineLayout,
                                    const vk::Buffer &vertexBuffer,
                                    const vk::Buffer &indexBuffer,
                                    const vk::CommandBuffer& commandBuffer, 
                                    uint32_t indicesCount,
                                    uint32_t imageIndex) {
    std::array<vk::ClearValue, 2> clearValues;
    clearValues[0].color =
        vk::ClearColorValue(std::array<float, 4>({{0.2f, 0.2f, 0.2f, 0.2f}}));
    clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderPassBeginInfo renderPassInfo(
        renderPass, m_vkSwapChainFramebuffers[imageIndex],
        vk::Rect2D(vk::Offset2D(0, 0), m_vkSwapChainExtent), clearValues);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               pipeline);

    commandBuffer.bindVertexBuffers(0, vertexBuffer, vk::DeviceSize(0));

    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
        m_vkDescriptorSets[m_nCurrentFrame], nullptr);

    commandBuffer.drawIndexed(indicesCount, 1, 0, 0,
                              0);

    commandBuffer.endRenderPass();

}

void VulkanRHI::createSyncObjects() {
    m_vkImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_vkInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo;

    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_vkImageAvailableSemaphores[i] =
            m_vkDevice.createSemaphore(semaphoreInfo);
        m_vkInFlightFences[i] = m_vkDevice.createFence(fenceInfo);
    }
}

bool VulkanRHI::BeginFrame() {
    if (m_frameActive) throw std::logic_error("Vulkan frame is already recording");
    if (!m_vkDevice || !m_fQueryFramebufferSizeCB ||
        m_vkInFlightFences.size() != MAX_FRAMES_IN_FLIGHT) {
        throw std::logic_error("Vulkan frame resources are not initialized");
    }
    uint32_t width = 0, height = 0;
    m_fQueryFramebufferSizeCB(width, height);
    if (width == 0 || height == 0) return false;
    if (!m_vkSwapChain || m_recreateSwapChain ||
        width != m_framebufferWidth || height != m_framebufferHeight) {
        recreateSwapChain();
    }
    if (!m_vkSwapChain) return false;
    const auto fence = m_vkInFlightFences[m_nCurrentFrame];
    if (m_vkDevice.waitForFences(fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("Vulkan frame fence wait failed");
    }
    const auto result = static_cast<vk::Result>(vkAcquireNextImageKHR(
        m_vkDevice, m_vkSwapChain, UINT64_MAX,
        m_vkImageAvailableSemaphores[m_nCurrentFrame], VK_NULL_HANDLE, &m_currentImage));
    if (result == vk::Result::eErrorOutOfDateKHR) {
        m_recreateSwapChain = true;
        return false;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Vulkan image acquisition failed: " + vk::to_string(result));
    }
    m_recreateSwapChain = result == vk::Result::eSuboptimalKHR;
    if (m_vkImagesInFlight[m_currentImage] &&
        m_vkDevice.waitForFences(m_vkImagesInFlight[m_currentImage], VK_TRUE,
                                 UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("Vulkan swapchain image fence wait failed");
    }
    auto command = m_vkCommandBuffers.at(m_nCurrentFrame);
    command.reset();
    command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    m_frameActive = true;
    return true;
}

vk::CommandBuffer VulkanRHI::GetFrameCommandBuffer() const {
    if (!m_frameActive) throw std::logic_error("No Vulkan frame is recording");
    return m_vkCommandBuffers.at(m_nCurrentFrame);
}

vk::Framebuffer VulkanRHI::GetFrameFramebuffer() const {
    if (!m_frameActive) throw std::logic_error("No Vulkan frame is recording");
    return m_vkSwapChainFramebuffers.at(m_currentImage);
}

void VulkanRHI::EndFrame() {
    const auto command = GetFrameCommandBuffer();
    command.end();
    const auto fence = m_vkInFlightFences[m_nCurrentFrame];
    const auto finished = m_vkRenderFinishedSemaphores.at(m_currentImage);
    const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    const vk::SubmitInfo submit(m_vkImageAvailableSemaphores[m_nCurrentFrame],
                                waitStage, command, finished);
    // Reset only after acquisition and recording have succeeded.
    m_vkDevice.resetFences(fence);
    m_vkGraphicsQueue.submit(submit, fence);
    m_vkImagesInFlight[m_currentImage] = fence;
    m_frameActive = false;
    m_nCurrentFrame = (m_nCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    const VkSemaphore semaphore = finished;
    const VkSwapchainKHR swapchain = m_vkSwapChain;
    const VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr,
                                  1, &semaphore, 1, &swapchain, &m_currentImage, nullptr};
    const auto result = static_cast<vk::Result>(vkQueuePresentKHR(m_vkPresentQueue, &present));
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        m_recreateSwapChain = true;
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Vulkan presentation failed: " + vk::to_string(result));
    }
}

void VulkanRHI::drawFrame(const vk::RenderPass &renderPass,
                          const vk::Pipeline &pipeline,
                          const vk::PipelineLayout &pipelineLayout,
                          const vk::Buffer &vertexBuffer,
                          const vk::Buffer &indexBuffer,
                          uint32_t indicesCount) {
    if (!BeginFrame()) return;
    recordCommandBuffer(renderPass, pipeline, pipelineLayout, vertexBuffer,
                         indexBuffer, GetFrameCommandBuffer(), indicesCount, m_currentImage);
    EndFrame();
}

void VulkanRHI::cleanupSwapChain() {
    for (auto framebuffer : m_vkSwapChainFramebuffers) {
        m_vkDevice.destroyFramebuffer(framebuffer);
    }
    m_vkSwapChainFramebuffers.clear();
    if (m_swapChainResourcesCreated && m_fDestroySwapChainCB) {
        m_fDestroySwapChainCB();
    }
    m_swapChainResourcesCreated = false;

    m_vkDevice.destroyImageView(m_vkColorImageView);
    m_vkDevice.destroyImage(m_vkColorImage);
    m_vkDevice.freeMemory(m_vkColorImageMemory);

    m_vkDevice.destroyImageView(m_vkDepthImageView);
    m_vkDevice.destroyImage(m_vkDepthImage);
    m_vkDevice.freeMemory(m_vkDepthImageMemory);

    m_vkColorImageView = nullptr;
    m_vkColorImage = nullptr;
    m_vkColorImageMemory = nullptr;
    m_vkDepthImageView = nullptr;
    m_vkDepthImage = nullptr;
    m_vkDepthImageMemory = nullptr;

    for (auto& imageView : m_vkSwapChainImageViews) {
        m_vkDevice.destroyImageView(imageView);
    }
    m_vkSwapChainImageViews.clear();

    m_vkDevice.destroySwapchainKHR(m_vkSwapChain);
    m_vkSwapChain = nullptr;
    m_vkSwapChainImages.clear();
    m_vkImagesInFlight.clear();
    for (auto semaphore : m_vkRenderFinishedSemaphores) {
        m_vkDevice.destroySemaphore(semaphore);
    }
    m_vkRenderFinishedSemaphores.clear();
}

void VulkanRHI::recreateSwapChain(const vk::RenderPass&) {
    uint32_t width = 0, height = 0;
    if (!m_fQueryFramebufferSizeCB) throw std::logic_error("Missing Vulkan framebuffer size callback");
    m_fQueryFramebufferSizeCB(width, height);
    if (width == 0 || height == 0) return;
    WaitGPU();
    cleanupSwapChain();
    createSwapChain();
}

uint32_t VulkanRHI::findMemoryType(uint32_t typeFilter,
                                   vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties =
        m_vkPhysicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            ((memProperties.memoryTypes[i].propertyFlags & properties) ==
             properties)) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanRHI::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                             vk::MemoryPropertyFlags properties,
                             vk::Buffer& buffer,
                             vk::DeviceMemory& bufferMemory) {
    if (size == 0 || buffer || bufferMemory) {
        throw std::invalid_argument("Vulkan buffer requires a nonzero size and empty output handles");
    }
    ScopedBuffer allocation(m_vkDevice);
    vk::BufferCreateInfo bufferInfo({}, size, usage);
    allocation.buffer = m_vkDevice.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements;
    memRequirements = m_vkDevice.getBufferMemoryRequirements(allocation.buffer);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    allocation.memory = m_vkDevice.allocateMemory(allocInfo);
    m_vkDevice.bindBufferMemory(allocation.buffer, allocation.memory, 0);
    buffer = allocation.buffer;
    bufferMemory = allocation.memory;
    allocation.Release();
}

void VulkanRHI::createVertexBuffer(void *buffer, vk::DeviceSize bufferSize, VulkanRHI::VertexBuffer &vertexBuffer) {
    if (!buffer) throw std::invalid_argument("Missing Vulkan vertex data");
    ScopedBuffer staging(m_vkDevice);
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 staging.buffer, staging.memory);

    // 上传数据
    void* data;
    data = m_vkDevice.mapMemory(staging.memory, 0, bufferSize);
    memcpy(data, buffer, (size_t)bufferSize);
    m_vkDevice.unmapMemory(staging.memory);

    // 创建设备专有缓冲区
    ScopedBuffer destination(m_vkDevice);
    if (vertexBuffer.buffer || vertexBuffer.heap) throw std::invalid_argument("Vulkan vertex output is not empty");
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eVertexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, destination.buffer,
                 destination.memory);
    copyBuffer(staging.buffer, destination.buffer, bufferSize);
    vertexBuffer.buffer = destination.buffer;
    vertexBuffer.heap = destination.memory;
    destination.Release();
}

void VulkanRHI::createIndexBuffer(void *buffer, vk::DeviceSize bufferSize, VulkanRHI::IndexBuffer &indexBuffer) {
    if (!buffer) throw std::invalid_argument("Missing Vulkan index data");
    ScopedBuffer staging(m_vkDevice);
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 staging.buffer, staging.memory);

    // 上传数据
    void* data;
    data = m_vkDevice.mapMemory(staging.memory, 0, bufferSize);
    memcpy(data, buffer, (size_t)bufferSize);
    m_vkDevice.unmapMemory(staging.memory);

    // 创建设备专有缓冲区
    ScopedBuffer destination(m_vkDevice);
    if (indexBuffer.buffer || indexBuffer.heap) throw std::invalid_argument("Vulkan index output is not empty");
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eIndexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, destination.buffer,
                 destination.memory);
    copyBuffer(staging.buffer, destination.buffer, bufferSize);
    indexBuffer.buffer = destination.buffer;
    indexBuffer.heap = destination.memory;
    destination.Release();
}

void VulkanRHI::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                           vk::DeviceSize size) {
    auto commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion(0, 0, size);
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
    const vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                                    vk::AccessFlagBits::eMemoryRead);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eAllCommands,
                                   {}, barrier, {}, {});

    endSingleTimeCommands(commandBuffer);
}

void VulkanRHI::copyBufferToImage(vk::Buffer srcBuffer, vk::Image image,
                                  uint32_t width, uint32_t height) {
    auto commandBuffer = beginSingleTimeCommands();

    vk::BufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = vk::Offset3D(0, 0, 0);
    region.imageExtent = vk::Extent3D(width, height, 1);

    commandBuffer.copyBufferToImage(
        srcBuffer, image, vk::ImageLayout::eTransferDstOptimal, region);

    endSingleTimeCommands(commandBuffer);
}

vk::DescriptorSetLayout VulkanRHI::createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding samplerLayoutBinding;
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType =
        vk::DescriptorType::eCombinedImageSampler;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};

    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);

    return m_vkDevice.createDescriptorSetLayout(layoutInfo);
}

void VulkanRHI::createUniformBuffers(std::vector<VulkanRHI::UniformBuffer> &uniformBuffers, vk::DeviceSize bufferSize) {
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent,
                     uniformBuffers[i].buffer, uniformBuffers[i].heap);
        uniformBuffers[i].size = bufferSize;
    }
}

void VulkanRHI::UpdateUniformBuffer(const std::vector<VulkanRHI::UniformBuffer> &uniformBuffers, void *ubo, size_t uboSize) {
    if (!ubo || uboSize == 0 || uniformBuffers.size() != MAX_FRAMES_IN_FLIGHT ||
        uniformBuffers[m_nCurrentFrame].size < uboSize) {
        throw std::invalid_argument("Invalid Vulkan uniform upload");
    }
    if (m_vkDevice.waitForFences(m_vkInFlightFences.at(m_nCurrentFrame), VK_TRUE,
                                 UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("Vulkan uniform upload fence wait failed");
    }

    // 上传数据
    void *data;
    data = m_vkDevice.mapMemory(uniformBuffers[m_nCurrentFrame].heap, 0, uboSize);
    memcpy(data, ubo, uboSize);
    m_vkDevice.unmapMemory(uniformBuffers[m_nCurrentFrame].heap);
}

void VulkanRHI::createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 2> poolSize;
    poolSize[0].type = vk::DescriptorType::eUniformBuffer;
    poolSize[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSize[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSize[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo poolInfo(
        {}, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT), poolSize);

    m_vkDescriptorPool = m_vkDevice.createDescriptorPool(poolInfo);
}

void VulkanRHI::createDescriptorSets(const vk::DescriptorSetLayout &descriptorSetLayout, 
                                     const std::vector<VulkanRHI::UniformBuffer> &uniformBuffers,
                                     const std::vector<VulkanRHI::Texture> &textures,
                                     const  vk::Sampler &textureSampler) {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = m_vkDescriptorPool;
    allocInfo.setSetLayouts(layouts);

    m_vkDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    m_vkDescriptorSets = m_vkDevice.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i].buffer, 0,
                                            uniformBuffers[i].size);

        vk::DescriptorImageInfo imageInfo(
            textureSampler, textures[0].descriptor,
            vk::ImageLayout::eShaderReadOnlyOptimal);

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites;
        descriptorWrites[0].dstSet = m_vkDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = m_vkDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType =
            vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        m_vkDevice.updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void VulkanRHI::getTextureFormat(const Image& img,
                                 vk::Format& internal_format) {
    if (img.compressed) {
        switch (img.compress_format) {
            case COMPRESSED_FORMAT::BC1:
                internal_format = vk::Format::eBc1RgbSrgbBlock;
                break;
            case COMPRESSED_FORMAT::BC1A:
                internal_format = vk::Format::eBc1RgbaSrgbBlock;
                break;
            case COMPRESSED_FORMAT::BC2:
                internal_format = vk::Format::eBc2SrgbBlock;
                break;
            case COMPRESSED_FORMAT::BC3:
                internal_format = vk::Format::eBc3SrgbBlock;
                break;
            case COMPRESSED_FORMAT::BC4:
                internal_format = (img.is_signed) ? vk::Format::eBc4SnormBlock
                                                  : vk::Format::eBc4UnormBlock;
                break;
            case COMPRESSED_FORMAT::BC5:
                internal_format = (img.is_signed) ? vk::Format::eBc5SnormBlock
                                                  : vk::Format::eBc5UnormBlock;
                break;
            case COMPRESSED_FORMAT::BC6H:
                internal_format = (img.is_signed)
                                      ? vk::Format::eBc6HSfloatBlock
                                      : vk::Format::eBc6HUfloatBlock;
                break;
            case COMPRESSED_FORMAT::BC7:
                internal_format = vk::Format::eBc7UnormBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_4x4:
                internal_format = vk::Format::eAstc4x4SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_5x4:
                internal_format = vk::Format::eAstc5x4SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_5x5:
                internal_format = vk::Format::eAstc5x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_6x5:
                internal_format = vk::Format::eAstc6x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_6x6:
                internal_format = vk::Format::eAstc6x6SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_8x5:
                internal_format = vk::Format::eAstc8x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_8x6:
                internal_format = vk::Format::eAstc8x6SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_8x8:
                internal_format = vk::Format::eAstc8x8SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_10x5:
                internal_format = vk::Format::eAstc10x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_10x6:
                internal_format = vk::Format::eAstc10x6SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_10x8:
                internal_format = vk::Format::eAstc10x8SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_10x10:
                internal_format = vk::Format::eAstc10x10SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_12x10:
                internal_format = vk::Format::eAstc12x10SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_12x12:
                internal_format = vk::Format::eAstc12x12SrgbBlock;
                break;
#if 0
            case COMPRESSED_FORMAT::ASTC_3x3x3:
                internal_format = vk::Format::eAstc3x3x3SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_4x3x3:
                internal_format = vk::Format::eAstc4x3x3SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_4x4x3:
                internal_format = vk::Format::eAstc4x4x3SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_4x4x4:
                internal_format = vk::Format::eAstc4x4x4SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_5x4x4:
                internal_format = vk::Format::eAstc5x4x4SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_5x5x4:
                internal_format = vk::Format::eAstc5x5x4SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_5x5x5:
                internal_format = vk::Format::eAstc5x5x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_6x5x5:
                internal_format = vk::Format::eAstc6x5x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_6x6x5:
                internal_format = vk::Format::eAstc6x6x5SrgbBlock;
                break;
            case COMPRESSED_FORMAT::ASTC_6x6x6:
                internal_format = vk::Format::eAstc6x6x6SrgbBlock;
                break;
#endif
            default:
                assert(0);
        }
    } else {
        if (img.bitcount == 8) {
            internal_format = vk::Format::eR8Unorm;
        } else if (img.bitcount == 16) {
            internal_format = vk::Format::eR16Unorm;
        } else if (img.bitcount == 24) {
            internal_format = vk::Format::eR8G8B8A8Unorm;
        } else if (img.bitcount == 32) {
            internal_format = vk::Format::eR8G8B8A8Unorm;
        } else if (img.bitcount == 64) {
            if (img.is_float) {
                internal_format = vk::Format::eR16G16B16A16Sfloat;
            } else {
                internal_format = vk::Format::eR16G16B16A16Snorm;
            }
        } else if (img.bitcount == 128) {
            if (img.is_float) {
                internal_format = vk::Format::eR32G32B32A32Sfloat;
            } else {
                internal_format = vk::Format::eR32G32B32A32Sint;
            }
        } else {
            assert(0);
        }
    }
}

void VulkanRHI::createImage(
    Image& image, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
    vk::SharingMode sharing_mode, vk::MemoryPropertyFlags properties,
    vk::Image& vk_image, vk::DeviceMemory& vk_image_memory,
    uint32_t queueFamilyIndexCount, uint32_t* queueFamilyIndices) {
    if (!image.data) {
        throw std::runtime_error("faild to load texture image!");
    }

    vk::Format format;
    getTextureFormat(image, format);
    createImage(image.Width, image.Height, vk::SampleCountFlagBits::e1, format,
                tiling, usage, sharing_mode, properties, vk_image,
                vk_image_memory, queueFamilyIndexCount, queueFamilyIndices);
}

void VulkanRHI::createImage(
    uint32_t width, uint32_t height, vk::SampleCountFlagBits numSamples,
    vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
    vk::SharingMode sharing_mode, vk::MemoryPropertyFlags properties,
    vk::Image& vk_image, vk::DeviceMemory& vk_image_memory,
    uint32_t queueFamilyIndexCount, uint32_t* queueFamilyIndices) {
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;

    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = sharing_mode;
    if (queueFamilyIndexCount > 0) {
        imageInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        imageInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    imageInfo.samples = numSamples;

    vk_image = m_vkDevice.createImage(imageInfo);

    vk::MemoryRequirements memRequirements =
        m_vkDevice.getImageMemoryRequirements(vk_image);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    vk_image_memory = m_vkDevice.allocateMemory(allocInfo);

    m_vkDevice.bindImageMemory(vk_image, vk_image_memory, 0);
}

VulkanRHI::Texture VulkanRHI::createTextureImage(Image& image) {
    VulkanRHI::Texture texture;
    vk::DeviceSize imageSize = static_cast<VkDeviceSize>(image.data_size);

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    // 上传数据
    void* data;
    data = m_vkDevice.mapMemory(stagingBufferMemory, 0, imageSize);
    memcpy(data, image.data, image.data_size);
    m_vkDevice.unmapMemory(stagingBufferMemory);

    auto indices = findQueueFamilies(m_vkPhysicalDevice, m_vkSurface);
    bool useSeperateTransferFamily =
        (indices.transferFamily.value() != indices.graphicsFamily.value());

    if (useSeperateTransferFamily) {
        std::array<uint32_t, 2> queueFamilyIndices = {
            indices.transferFamily.value(), indices.graphicsFamily.value()};
        createImage(image, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferDst |
                        vk::ImageUsageFlagBits::eSampled,
                    vk::SharingMode::eConcurrent,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, texture.image,
                    texture.heap,
                    static_cast<uint32_t>(queueFamilyIndices.size()),
                    queueFamilyIndices.data());
    } else {
        std::array<uint32_t, 1> queueFamilyIndices = {
            indices.graphicsFamily.value()};
        createImage(image, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferDst |
                        vk::ImageUsageFlagBits::eSampled,
                    vk::SharingMode::eExclusive,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, texture.image,
                    texture.heap,
                    static_cast<uint32_t>(queueFamilyIndices.size()),
                    queueFamilyIndices.data());
    }

    vk::Format format;
    getTextureFormat(image, format);

    transitionImageLayout(texture.image, format, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);

    copyBufferToImage(stagingBuffer, texture.image, image.Width,
                      image.Height);

    transitionImageLayout(texture.image, format,
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);

    m_vkDevice.destroyBuffer(stagingBuffer);
    m_vkDevice.freeMemory(stagingBufferMemory);

    texture.descriptor = createImageView(texture.image, format,
                                           vk::ImageAspectFlagBits::eColor);

    return texture;
}

vk::ImageView VulkanRHI::createImageView(vk::Image image, vk::Format format,
                                         vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vk::ImageView imageView;
    imageView = m_vkDevice.createImageView(viewInfo);

    return imageView;
}

vk::Sampler VulkanRHI::createTextureSampler() {
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = VK_TRUE;

    vk::PhysicalDeviceProperties properties =
        m_vkPhysicalDevice.getProperties();
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    return m_vkDevice.createSampler(samplerInfo);
}

vk::CommandBuffer VulkanRHI::beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo(
        m_vkCommandPoolTransfer, vk::CommandBufferLevel::ePrimary, 1);

    vk::CommandBuffer commandBuffer;
    commandBuffer = m_vkDevice.allocateCommandBuffers(allocInfo).front();

    vk::CommandBufferBeginInfo beginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer.begin(beginInfo);

    return commandBuffer;
}

void VulkanRHI::endSingleTimeCommands(vk::CommandBuffer& commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(commandBuffer);

    const auto fence = m_vkDevice.createFenceUnique({});
    m_vkTransferQueue.submit(submitInfo, *fence);
    if (m_vkDevice.waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("Vulkan upload fence wait failed");
    }

    m_vkDevice.freeCommandBuffers(m_vkCommandPoolTransfer, commandBuffer);
}

void VulkanRHI::transitionImageLayout(vk::Image image, vk::Format format,
                                      vk::ImageLayout oldLayout,
                                      vk::ImageLayout newLayout,
                                      uint32_t srcQueueFamilyIndex,
                                      uint32_t dstQueueFamilyIndex) {
    auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
    barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader |
                           vk::PipelineStageFlagBits::eVertexShader |
                           vk::PipelineStageFlagBits::eComputeShader;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage,
                                  vk::DependencyFlagBits::eByRegion, nullptr,
                                  nullptr, barrier);

    endSingleTimeCommands(commandBuffer);
}

vk::Format VulkanRHI::findSupportedFormat(
    const std::vector<vk::Format>& candidates, vk::ImageTiling tiling,
    vk::FormatFeatureFlags features) {
    for (auto format : candidates) {
        vk::FormatProperties props =
            m_vkPhysicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal &&
                   (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

vk::Format VulkanRHI::findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

static bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint ||
           format == vk::Format::eD24UnormS8Uint;
}

void VulkanRHI::createDepthResources() {
    auto depthFormat = findDepthFormat();

    createImage(m_vkSwapChainExtent.width, m_vkSwapChainExtent.height,
                m_vkMsaaSamples, depthFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::SharingMode::eExclusive,
                vk::MemoryPropertyFlagBits::eDeviceLocal, m_vkDepthImage,
                m_vkDepthImageMemory);
    m_vkDepthImageView = createImageView(m_vkDepthImage, depthFormat,
                                         vk::ImageAspectFlagBits::eDepth);
}

vk::SampleCountFlagBits VulkanRHI::getMaxUsableSampleCount() {
    vk::PhysicalDeviceProperties physicalDeviceProperties =
        m_vkPhysicalDevice.getProperties();

    vk::SampleCountFlags counts =
        physicalDeviceProperties.limits.framebufferColorSampleCounts &
        physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    const auto requested = m_fGetGfxConfigCB ? m_fGetGfxConfigCB().msaaSamples : 1;
    if (requested == 0 || requested > 64 || (requested & (requested - 1)) != 0) {
        throw std::invalid_argument("Vulkan MSAA count must be a power of two in [1,64]");
    }
    if (requested >= 64 && (counts & vk::SampleCountFlagBits::e64)) {
        return vk::SampleCountFlagBits::e64;
    }
    if (requested >= 32 && (counts & vk::SampleCountFlagBits::e32)) {
        return vk::SampleCountFlagBits::e32;
    }
    if (requested >= 16 && (counts & vk::SampleCountFlagBits::e16)) {
        return vk::SampleCountFlagBits::e16;
    }
    if (requested >= 8 && (counts & vk::SampleCountFlagBits::e8)) {
        return vk::SampleCountFlagBits::e8;
    }
    if (requested >= 4 && (counts & vk::SampleCountFlagBits::e4)) {
        return vk::SampleCountFlagBits::e4;
    }
    if (requested >= 2 && (counts & vk::SampleCountFlagBits::e2)) {
        return vk::SampleCountFlagBits::e2;
    }

    return vk::SampleCountFlagBits::e1;
}

void VulkanRHI::createColorResources() {
    if (m_vkMsaaSamples == vk::SampleCountFlagBits::e1) return;
    vk::Format colorFormat = m_vkSwapChainImageFormat;

    createImage(m_vkSwapChainExtent.width, m_vkSwapChainExtent.height,
                m_vkMsaaSamples, colorFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment,
                vk::SharingMode::eExclusive,
                vk::MemoryPropertyFlagBits::eDeviceLocal, m_vkColorImage,
                m_vkColorImageMemory);
    m_vkColorImageView = createImageView(m_vkColorImage, colorFormat,
                                         vk::ImageAspectFlagBits::eColor);
}

void VulkanRHI::CreateSwapChain() {
    if (!m_fCreateSwapChainCB) throw std::logic_error("Missing Vulkan swapchain resource callback");
    if (!m_vkSwapChain) return;
    m_swapChainResourcesCreated = true;
    m_fCreateSwapChainCB();
}
