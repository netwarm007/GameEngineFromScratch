#include "VulkanRHI.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <vector>

const int MAX_FRAMES_IN_FLIGHT = 2;

static std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

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
VulkanRHI::VulkanRHI() {
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::cout << extensionCount << " extensions supported\n";

    std::vector<VkExtensionProperties> extensions(extensionCount);

    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "available extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << '\t' << extension.extensionName << std::endl;
        if (strncmp(extension.extensionName, "VK_KHR_portability_subset", sizeof(VK_MAX_EXTENSION_NAME_SIZE)) == 0) {
            deviceExtensions.push_back("VK_KHR_portability_subset");
        }
    }
}

VulkanRHI::~VulkanRHI() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(m_vkDevice, m_vkImageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(m_vkDevice, m_vkRenderFinishedSemaphores[i], nullptr);
        vkDestroyFence(m_vkDevice, m_vkInFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(m_vkDevice, m_vkCommandPool, nullptr);

    vkDestroyPipeline(m_vkDevice, m_vkGraphicPipeline, nullptr);

    vkDestroyPipelineLayout(m_vkDevice, m_vkPipelineLayout, nullptr);

    for (auto& framebuffer : m_vkSwapChainFramebuffers) {
        vkDestroyFramebuffer(m_vkDevice, framebuffer, nullptr);
    }

    vkDestroyRenderPass(m_vkDevice, m_vkRenderPass, nullptr);

    for (auto& imageView : m_vkSwapChainImageViews) {
        vkDestroyImageView(m_vkDevice, imageView, nullptr);
    }

    vkDestroySwapchainKHR(m_vkDevice, m_vkSwapChain, nullptr);

    vkDestroyDevice(m_vkDevice, nullptr);     // 销毁逻辑设备

    vkDestroySurfaceKHR(m_vkInstance, m_vkSurface, nullptr);

    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugMessenger, nullptr);
    }

    vkDestroyInstance(m_vkInstance, nullptr); // 会在销毁instance的同时销毁物理设备对象
}

void VulkanRHI::createInstance () {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
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

    auto extensions = getRequiredExtensions();

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

void VulkanRHI::createSurface() {

}

void VulkanRHI::pickPhysicalDevice() {

}

void VulkanRHI::createLogicalDevice () {

}

void VulkanRHI::createSwapChain () {

}

void VulkanRHI::createImageViews() {

}

void VulkanRHI::createRenderPass() {

}

void VulkanRHI::createGraphicsPipeline() {

}

void VulkanRHI::createFramebuffers() {

}

void VulkanRHI::createCommandPool() {

}

void VulkanRHI::createSyncObjects() {

}
