#include "config.h"

#include "AssetLoader.hpp"
#include "D3d12Application.hpp"
#include "PVR.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace My;

int main() {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "DirectX 12 RHI Test");
    D3d12Application app(config);

    AssetLoader asset_loader;

    assert(asset_loader.Initialize() == 0 && "Asset Loader Initialize Failed!");

    // 创建窗口
    {
        assert(app.Initialize() == 0 && "App Initialize Failed!");

        app.CreateMainWindow();
    }

    auto& rhi = app.GetRHI();

    // 创建图形管道
    {
        auto vertShader =
            asset_loader.SyncOpenAndReadBinary("Shaders/HLSL/simple.vert.cso");
        auto fragShader =
            asset_loader.SyncOpenAndReadBinary("Shaders/HLSL/simple.frag.cso");
        rhi.setShaders(vertShader, fragShader);

        // 创建 Descriptor 布局
        rhi.CreateDescriptorSetLayout();

        // 创建 PSO
        rhi.CreateGraphicsPipeline();
    }

    // 创建命令清单池
    rhi.CreateCommandPools();

    // 创建命令列表
    rhi.CreateCommandLists();

    // 加载贴图
    {
        auto buf =
            asset_loader.SyncOpenAndReadBinary("Textures/viking_room.pvr");
        PVR::PvrParser parser;
        auto image = parser.Parse(buf);
        auto res_id = rhi.CreateTextureImage(image);
    }

    // 创建采样器
    rhi.CreateTextureSampler();

    // 加载模型
    {
        auto model_path =
            asset_loader.GetFileRealPath("Models/viking_room.obj");

        assert(!model_path.empty() && "Can not find model file!");

        std::vector<D3d12RHI::Vertex> vertices;
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
                D3d12RHI::Vertex vertex{};

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
    rhi.CreateVertexBuffer();

    // 创建索引缓冲区
    rhi.CreateIndexBuffer();

    // 创建常量缓冲区
    rhi.CreateUniformBuffers();

    // 创建资源描述子池
    rhi.CreateDescriptorPool();

    // 创建资源描述子集
    rhi.CreateDescriptorSets();

    // 主消息循环
    while (!app.IsQuit()) {
        app.Tick();

        // 绘制一帧
        rhi.DrawFrame();
    }

    app.Finalize();

    return 0;
}