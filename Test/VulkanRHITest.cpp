#include "AssetLoader.hpp"
#include "config.h"
#if defined(OS_MACOS)
#include "CocoaVulkanApplication.h"
#endif
#if defined(OS_WINDOWS)
#include "VulkanApplication.hpp"
#endif
#include "PVR.hpp"
#include "Vulkan/VulkanRHI.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace My;

int main() {
    try {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "Vulkan RHI Test");
#if defined(OS_MACOS)
        CocoaVulkanApplication app(config);
#endif
#if defined(OS_WINDOWS)
        VulkanApplication app(config);
#endif

        AssetLoader asset_loader;

        assert(asset_loader.Initialize() == 0 &&
               "Asset Loader Initialize Failed!");

        // 创建窗口
        {
            assert(app.Initialize() == 0 && "App Initialize Failed!");

            app.CreateMainWindow();
        }

        // 获取窗口所需的Extensions
#if defined(OS_MACOS)
        std::vector<const char*> extensions = {
            "VK_KHR_surface", "VK_EXT_metal_surface",
            "VK_KHR_portability_enumeration",
            "VK_KHR_get_physical_device_properties2"};
#endif
#if defined(OS_WINDOWS)
        std::vector<const char*> extensions = {"VK_KHR_surface",
                                               "VK_KHR_win32_surface"};
#endif

        VulkanRHI rhi;
        // 设置回调函数
        auto getFramebufferSize = [&app](int& width, int& height) {
            app.GetFramebufferSize(width, height);
        };

        // 设置回调函数
        rhi.setFramebufferSizeQueryCB(getFramebufferSize);

        // 创建实例
        rhi.createInstance(extensions);

        // 开启调试层（对发行版不起作用）
        rhi.setupDebugMessenger();

        // 创建（连接）画布
        {
            auto createWindowSurface = [&app](const vk::Instance& instance,
                                              vk::SurfaceKHR& surface) {
                VkSurfaceKHR _surface;
                assert(instance);

                if (app.CreateWindowSurface(instance, _surface) != VK_SUCCESS) {
                    throw std::runtime_error("faild to create window surface!");
                }

                surface = _surface;
            };
            rhi.createSurface(createWindowSurface);
        }

        // 枚举物理设备并选择
        rhi.pickPhysicalDevice();

        // 创建逻辑设备
        rhi.createLogicalDevice();

        // 创建 SwapChain
        rhi.createSwapChain();

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
            auto vertShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/simple.vert.spv");
            auto fragShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/simple.frag.spv");
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
            auto buf =
                asset_loader.SyncOpenAndReadBinary("Textures/viking_room.pvr");
            PVR::PvrParser parser;
            auto image = parser.Parse(buf);
            rhi.createTextureImage(image);
            rhi.createTextureImageView(image);
        }

        // 创建采样器
        rhi.createTextureSampler();

        // 加载模型
        {
            auto model_path =
                asset_loader.GetFileRealPath("Models/viking_room.obj");

            assert(!model_path.empty() && "Can not find model file!");

            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                  model_path.c_str())) {
                throw std::runtime_error(warn + err);
            }

            for (const auto& shape : shapes) {
                for (const auto& index : shape.mesh.indices) {
                    Vertex vertex{};

                    vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                                  attrib.vertices[3 * index.vertex_index + 1],
                                  attrib.vertices[3 * index.vertex_index + 2]};

                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

                    vertex.color = {1.0f, 1.0f, 1.0f};

                    vertices.push_back(vertex);
                    indices.push_back(indices.size());
                }

                rhi.setModel(vertices, indices);
            }
        }

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
        while (!app.IsQuit()) {
            app.Tick();
            rhi.drawFrame();
        }

        // 销毁Vulkan上下文
        rhi.destroyAll();

        app.Finalize();
    } catch (vk::SystemError& err) {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return (-1);
    } catch (std::exception& err) {
        std::cout << "std::exception: " << err.what() << std::endl;
        return (-1);
    } catch (...) {
        std::cout << "unknown error\n";
        return (-1);
    }

    return 0;
}