#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "AssetLoader.hpp"
#include "Vulkan/VulkanRHI.hpp"

using namespace My;

int main() {
    // 创建窗口
    GLFWwindow* window;
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(800, 600, "Vulkan Window", nullptr, nullptr);
    }

    // 获取窗口所需的Extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VulkanRHI rhi;

    // 创建实例
    rhi.createInstance(extensions);

    // 开启调试层（对发行版不起作用）
    rhi.setupDebugMessenger();

    // 创建（连接）画布
    {
        auto createWindowSurface = [window](const VkInstance& instance, VkSurfaceKHR& surface) {
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
                throw std::runtime_error("faild to create window surface!");
            }
        };
        rhi.createSurface(createWindowSurface);
    }

    // 枚举物理设备并选择
    rhi.pickPhysicalDevice();

    // 创建逻辑设备
    rhi.createLogicalDevice();

    // 创建 SwapChain
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        rhi.createSwapChain(width, height);
    }

    // 创建Image View
    rhi.createImageViews();

    // 获取命令队列
    rhi.getDeviceQueues();

    // 创建渲染工序（Render Pass）
    rhi.createRenderPass();

    // 创建图形管道
    {
        AssetLoader asset_loader;
        auto vertShader = asset_loader.SyncOpenAndReadBinary("Shaders/Vulkan/simple_v.spv");
        auto fragShader = asset_loader.SyncOpenAndReadBinary("Shaders/Vulkan/simple_f.spv");
        rhi.createGraphicsPipeline(vertShader, fragShader);
    }

    // 创建 Framebuffers
    rhi.createFramebuffers();

    // 创建 Command Pools
    rhi.createCommandPool();

    // 创建 Command Buffer
    rhi.createCommandBuffers();
    
    // 创建同步对象
    rhi.createSyncObjects();

    // 主消息循环
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        rhi.drawFrame();
    }

    return 0;
}