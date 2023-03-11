#include "config.h"

#include "AssetLoader.hpp"
#include "D3d12Application.hpp"
#include "PVR.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace My;

template <class T>
inline void SafeRelease(T** ppInterfaceToRelease) {
    if (*ppInterfaceToRelease != nullptr) {
        (*ppInterfaceToRelease)->Release();

        (*ppInterfaceToRelease) = nullptr;
    }
};

struct Vertex {
    Vector3f pos;
    Vector3f color;
    Vector2f texCoord;
};

struct UniformBufferObject {
    Matrix4X4f model;
    Matrix4X4f view;
    Matrix4X4f proj;
};

constexpr size_t uboSize = ALIGN(sizeof(UniformBufferObject), 256);

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

    // 创建并登记资源创建回调函数
    ID3D12PipelineState* pPipelineState;
    ID3D12RootSignature* pRootSignature;
    ID3D12Resource* pTexture;
    std::vector<D3d12RHI::IndexBuffer> IndexBuffers;
    std::vector<D3d12RHI::VertexBuffer> VertexBuffers;

    D3d12RHI::CreateResourceFunc createResourceFunc = [&rhi, &pPipelineState,
                                                       &pRootSignature,
                                                       &pTexture, &IndexBuffers,
                                                       &VertexBuffers,
                                                       &asset_loader,
                                                       config]() {
        // 创建图形管道
        {
            D3D12_SHADER_BYTECODE vertexShaderModule;
            D3D12_SHADER_BYTECODE pixelShaderModule;

            auto vertShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/HLSL/simple.vert.cso");
            auto fragShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/HLSL/simple.frag.cso");

            vertexShaderModule.BytecodeLength = vertShader.GetDataSize();
            vertexShaderModule.pShaderBytecode = vertShader.MoveData();

            pixelShaderModule.BytecodeLength = fragShader.GetDataSize();
            pixelShaderModule.pShaderBytecode = fragShader.MoveData();

            // 创建 Descriptor 布局
            pRootSignature = rhi.CreateRootSignature(pixelShaderModule);

            // 创建 PSO
            // create rasterizer descriptor
            D3D12_RASTERIZER_DESC rsd{
                D3D12_FILL_MODE_SOLID,
                D3D12_CULL_MODE_BACK,
                TRUE,
                D3D12_DEFAULT_DEPTH_BIAS,
                D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                TRUE,
                FALSE,
                FALSE,
                0,
                D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF};

            const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlend{
                FALSE,
                FALSE,
                D3D12_BLEND_ONE,
                D3D12_BLEND_ZERO,
                D3D12_BLEND_OP_ADD,
                D3D12_BLEND_ONE,
                D3D12_BLEND_ZERO,
                D3D12_BLEND_OP_ADD,
                D3D12_LOGIC_OP_NOOP,
                D3D12_COLOR_WRITE_ENABLE_ALL};

            const D3D12_BLEND_DESC bld{FALSE,
                                       FALSE,
                                       {
                                           defaultRenderTargetBlend,
                                           defaultRenderTargetBlend,
                                           defaultRenderTargetBlend,
                                           defaultRenderTargetBlend,
                                           defaultRenderTargetBlend,
                                           defaultRenderTargetBlend,
                                           defaultRenderTargetBlend,
                                       }};

            static const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp{
                D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};

            D3D12_DEPTH_STENCIL_DESC dsd{TRUE,
                                         D3D12_DEPTH_WRITE_MASK_ALL,
                                         D3D12_COMPARISON_FUNC_LESS,
                                         FALSE,
                                         D3D12_DEFAULT_STENCIL_READ_MASK,
                                         D3D12_DEFAULT_STENCIL_WRITE_MASK,
                                         defaultStencilOp,
                                         defaultStencilOp};

            // create the input layout object
            static const D3D12_INPUT_ELEMENT_DESC ied_simple[]{
                {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
                 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 0,
                 offsetof(Vertex, texCoord),
                 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

            // describe and create the graphics pipeline state object (PSO)
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psod{};
            psod.VS = vertexShaderModule;
            psod.PS = pixelShaderModule;
            psod.BlendState = bld;
            psod.SampleMask = UINT_MAX;
            psod.RasterizerState = rsd;
            psod.DepthStencilState = dsd;
            psod.InputLayout = {ied_simple, _countof(ied_simple)};

            psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            psod.NumRenderTargets = 1;
            psod.SampleDesc.Count = config.msaaSamples;
            if (config.msaaSamples > 1) {
                psod.SampleDesc.Quality =
                    DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
            } else {
                psod.SampleDesc.Quality = 0;
            }
            psod.DSVFormat = ::DXGI_FORMAT_D32_FLOAT;
            psod.RTVFormats[0] = ::DXGI_FORMAT_R8G8B8A8_UNORM;

            psod.pRootSignature = pRootSignature;

            pPipelineState = rhi.CreateGraphicsPipeline(psod);
        }

        // 加载贴图
        {
            auto buf =
                asset_loader.SyncOpenAndReadBinary("Textures/viking_room.pvr");
            PVR::PvrParser parser;
            auto image = parser.Parse(buf);
            pTexture = rhi.CreateTextureImage(image);
        }

        // 创建采样器
        rhi.CreateTextureSampler();

        // 加载模型
        {
            auto model_path =
                asset_loader.GetFileRealPath("Models/viking_room.obj");

            assert(!model_path.empty() && "Can not find model file!");

            using Index = uint32_t;
            std::vector<Vertex> vertices;
            std::vector<Index> indices;
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
            }

            // 创建顶点缓冲区
            VertexBuffers.emplace_back(rhi.CreateVertexBuffer(
                vertices.data(), vertices.size(), sizeof(Vertex)));

            // 创建索引缓冲区
            IndexBuffers.emplace_back(rhi.CreateIndexBuffer(
                indices.data(), indices.size(), sizeof(Index)));
        }

        // 创建常量缓冲区
        rhi.CreateUniformBuffers(uboSize);

        // 创建资源描述子池
        rhi.CreateDescriptorPool(2, L"CbvSrvUav Heap",
                                 config.kMaxInFlightFrameCount);

        // 创建资源描述子集
        rhi.CreateDescriptorSets(uboSize, &pTexture, 1);
    };

    rhi.CreateResourceCB(createResourceFunc);

    // 创建并登记资源销毁回调函数
    D3d12RHI::DestroyResourceFunc destroyResourceFunc =
        [&pTexture, &pPipelineState, &pRootSignature, &VertexBuffers,
         &IndexBuffers]() {
            SafeRelease(&pPipelineState);
            SafeRelease(&pRootSignature);

            SafeRelease(&pTexture);

            for (auto& buf : VertexBuffers) {
                SafeRelease(&buf.buffer);
            }

            for (auto& buf : IndexBuffers) {
                SafeRelease(&buf.buffer);
            }
        };

    rhi.DestroyResourceCB(destroyResourceFunc);

    // 创建图形资源
    rhi.CreateGraphicsResources();

    Vector4f clearColor{0.3f, 0.3f, 0.3f, 1.0f};

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

        rhi.UpdateUniformBufer(&ubo, uboSize);

        // 绘制一帧
        rhi.BeginFrame();
        rhi.BeginPass(clearColor);
        rhi.SetPipelineState(pPipelineState);
        rhi.SetRootSignature(pRootSignature);
        rhi.Draw(VertexBuffers[0].descriptor, IndexBuffers[0].descriptor,
                 D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                 IndexBuffers[0].indexCount);
        rhi.EndPass();
        rhi.EndFrame();

        rhi.Present();
    }

    app.Finalize();

    return 0;
}