#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "AssetLoader.hpp"
#include "PNG.hpp"
#include "Vulkan/VulkanRHI.hpp"

using namespace My;

int main() {
    // 创建窗口
    GLFWwindow* window;
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(800, 600, "GEFS Vulkan Window", nullptr, nullptr);
    }

    // 获取窗口所需的Extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VulkanRHI rhi;
    // 设置回调函数
    auto getFramebufferSize = [window](int& width, int& height) {
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
    };

    rhi.setFramebufferSizeQueryCB(getFramebufferSize);

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
        rhi.createSwapChain();
    }

    // 创建Image View
    rhi.createImageViews();

    // 获取命令队列
    rhi.getDeviceQueues();

    // 创建渲染工序（Render Pass）
    rhi.createRenderPass();

    // 创建 Descriptor 布局
    rhi.createDescriptorSetLayout();

    // 创建图形管道
    {
        AssetLoader asset_loader;
        auto vertShader = asset_loader.SyncOpenAndReadBinary("Shaders/Vulkan/simple.vert.spv");
        auto fragShader = asset_loader.SyncOpenAndReadBinary("Shaders/Vulkan/simple.frag.spv");
        rhi.setShaders(vertShader, fragShader);
        rhi.createGraphicsPipeline();
    }

    // 创建 Command Pools
    rhi.createCommandPool();

    // 创建backend RT
    rhi.createColorResources();

    // 创建深度缓冲区
    rhi.createDepthResources();

    // 创建 Framebuffers
    rhi.createFramebuffers();

    // 加载贴图
    {
        AssetLoader asset_loader;
        auto buf = asset_loader.SyncOpenAndReadBinary("Textures/Lava_03_emissive-1K.png");
        PngParser parser;
        auto image = parser.Parse(buf);
        rhi.createTextureImage(image);
        rhi.createTextureImageView(image);
    }

    // 创建采样器
    rhi.createTextureSampler();

    // 创建顶点缓冲区
    rhi.createVertexBuffer();

    // 创建索引缓冲区
    rhi.createIndexBuffer();

    // 创建常量缓冲区
    rhi.createUniformBuffers();

    // 创建资源描述子池
    rhi.createDescriptorPool();

    // 创建资源描述子集
    rhi.createDescriptorSets();

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