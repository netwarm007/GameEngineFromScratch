#include "AssetLoader.hpp"
#include "PVR.hpp"
#include "Vulkan/VulkanRHI.hpp"
#include "VulkanApplication.hpp"
#include "config.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace My;

struct Vertex {
    Vector3f pos;
    Vector2f texCoord;

    static constexpr vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription;
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, 2>
    getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2>
            attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    Matrix4X4f model;
    Matrix4X4f view;
    Matrix4X4f proj;
};

int main() {
    try {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "Vulkan RHI Test");
        VulkanApplication app(config);

        AssetLoader asset_loader;

        assert(asset_loader.Initialize() == 0 &&
               "Asset Loader Initialize Failed!");

        // 创建窗口
        {
            assert(app.Initialize() == 0 && "App Initialize Failed!");

            app.CreateMainWindow();
        }

        auto& rhi = app.GetRHI();

        // 获取命令队列
        rhi.getDeviceQueues();

        std::vector<VulkanRHI::Shader> vkShaders;

        vk::RenderPass vkRenderPass;
        vk::Pipeline vkGraphicPipeline;
        vk::DescriptorSetLayout vkDescriptorSetLayout;
        vk::PipelineLayout vkPipelineLayout;
        vk::Sampler vkTextureSampler;

        std::vector<VulkanRHI::Texture> textures;
        std::vector<VulkanRHI::IndexBuffer> indexBuffers;
        std::vector<VulkanRHI::VertexBuffer> vertexBuffers;
        std::vector<VulkanRHI::UniformBuffer> uniformBuffers;

        // 创建 Descriptor 布局
        vkDescriptorSetLayout = rhi.createDescriptorSetLayout();

        // 创建图形管道
        {
            auto vertShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/simple.vert.spv");
            auto fragShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/simple.frag.spv");
            vkShaders.emplace_back(rhi.createShaderModule(vertShader), vk::ShaderStageFlagBits::eVertex, "simple_vert_main");
            vkShaders.emplace_back(rhi.createShaderModule(fragShader), vk::ShaderStageFlagBits::eFragment, "simple_frag_main");
        }

        // 加载贴图
        {
            auto buf = asset_loader.SyncOpenAndReadBinary(
                "Textures/viking_room.pvr");
            PVR::PvrParser parser;
            auto image = parser.Parse(buf);
            textures.emplace_back(rhi.createTextureImage(image));
        }

        // 创建采样器
        vkTextureSampler = rhi.createTextureSampler();

        // 加载模型
        {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            auto model_path =
                asset_loader.GetFileRealPath("Models/viking_room.obj");

            assert(!model_path.empty() && "Can not find model file!");

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn,
                                    &err, model_path.c_str())) {
                throw std::runtime_error(warn + err);
            }

            for (const auto& shape : shapes) {
                for (const auto& index : shape.mesh.indices) {
                    Vertex vertex{};

                    vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]};

                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f -
                            attrib.texcoords[2 * index.texcoord_index +
                                                1]};

                    vertices.push_back(vertex);
                    indices.push_back(indices.size());
                }
            }

            // 创建顶点缓冲区
            vertexBuffers.resize(1);
            rhi.createVertexBuffer(vertices.data(), sizeof(vertices[0]) * vertices.size(), vertexBuffers[0]);

            // 创建索引缓冲区
            indexBuffers.resize(1);
            rhi.createIndexBuffer(indices.data(), sizeof(indices[0]) * indices.size(), indexBuffers[0]);
            indexBuffers[0].indexCount = indices.size();
        }


        // 创建常量缓冲区
        rhi.createUniformBuffers(uniformBuffers, sizeof(UniformBufferObject));

        // 创建资源描述子池
        rhi.createDescriptorPool();

        // 创建资源描述子集
        rhi.createDescriptorSets(vkDescriptorSetLayout, uniformBuffers, textures, vkTextureSampler);

        VulkanRHI::CreateSwapChainCBFunc createSwapChainCBFunc =
            [&rhi, &vkShaders, &vkGraphicPipeline, &vkPipelineLayout,
             &vkRenderPass, &vkDescriptorSetLayout]() {
                // 创建渲染工序（Render Pass）
                vkRenderPass = rhi.createRenderPass();

                // 创建图形管道
                {
                    auto bindingDesc = Vertex::getBindingDescription();
                    auto attrDescs = Vertex::getAttributeDescriptions();
                    rhi.createGraphicsPipeline(
                        vkShaders.data(), vkShaders.size(), bindingDesc,
                        attrDescs.data(), attrDescs.size(),
                        vkDescriptorSetLayout, vkRenderPass, vkPipelineLayout,
                        vkGraphicPipeline);
                }

                // 创建backend RT
                rhi.createColorResources();

                // 创建深度缓冲区
                rhi.createDepthResources();

                // 创建 Framebuffers
                rhi.createFramebuffers(vkRenderPass);

            };

        rhi.CreateSwapChainCB(createSwapChainCBFunc);

        // 创建并登记资源销毁回调函数
        VulkanRHI::DestroySwapChainCBFunc destroySwapChainCBFunc =
            [&rhi, &vkGraphicPipeline, &vkPipelineLayout,
             &vkRenderPass]() {
                auto device = rhi.GetDevice();

                // 释放图形管道
                device.destroyPipeline(vkGraphicPipeline);

                // 释放管道布局
                device.destroyPipelineLayout(vkPipelineLayout);

                // 释放Render Pass
                device.destroyRenderPass(vkRenderPass);
            };

        rhi.DestroySwapChainCB(destroySwapChainCBFunc);

        // 创建图形渲染交换链
        rhi.CreateSwapChain();

        // 主消息循环
        while (!app.IsQuit()) {
            app.Tick();

            static auto startTime = std::chrono::high_resolution_clock::now();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(
                            currentTime - startTime)
                            .count();

            UniformBufferObject ubo{};
            BuildIdentityMatrix(ubo.model);
            MatrixRotationAxis(ubo.model, {0.0f, 0.0f, 1.0f}, time * PI / 8.0f);
            BuildViewRHMatrix(ubo.view, {2.0f, 2.0f, 2.0f}, {0.0f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 1.0f});
            uint32_t width, height;
            app.GetFramebufferSize(width, height);
            BuildPerspectiveFovRHMatrix(ubo.proj, PI / 4.0f, width / (float)height,
                                        0.1f, 10.0f);
            ubo.proj[1][1] *= -1.0f;

            rhi.UpdateUniformBuffer(uniformBuffers, &ubo, sizeof(ubo));

            rhi.drawFrame(vkRenderPass, vkGraphicPipeline, vkPipelineLayout, vertexBuffers[0].buffer, indexBuffers[0].buffer, indexBuffers[0].indexCount);
        }

        auto device = rhi.GetDevice();

        // 释放着色器资源描述子布局
        device.destroyDescriptorSetLayout(vkDescriptorSetLayout);

        // 释放着色器模块
        std::for_each(vkShaders.begin(), vkShaders.end(),
                        [&device](auto& shader) {
                            device.destroyShaderModule(shader.module);
                        });
        vkShaders.clear();

        // 释放贴图采样器
        device.destroySampler(vkTextureSampler);

        // 释放贴图
        std::for_each(textures.begin(), textures.end(),
                        [&device](VulkanRHI::Texture& tex) {
                            device.destroyImageView(tex.descriptor);
                            device.destroyImage(tex.image);
                            device.freeMemory(tex.heap);
                        });
        textures.clear();

        // 释放Uniform Buffer
        std::for_each(uniformBuffers.begin(), uniformBuffers.end(),
                        [&device](VulkanRHI::UniformBuffer& ubo) {
                            device.destroyBuffer(ubo.buffer);
                            device.freeMemory(ubo.heap);
                        });
        uniformBuffers.clear();

        // 释放索引缓冲区
        std::for_each(indexBuffers.begin(), indexBuffers.end(),
                        [&device](auto& idxBuf) {
                            device.destroyBuffer(idxBuf.buffer);
                            device.freeMemory(idxBuf.heap);
                        });
        indexBuffers.clear();

        // 释放顶点缓冲区
        std::for_each(vertexBuffers.begin(), vertexBuffers.end(),
                        [&device](auto& vertBuf) {
                            device.destroyBuffer(vertBuf.buffer);
                            device.freeMemory(vertBuf.heap);
                        });
        vertexBuffers.clear();

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