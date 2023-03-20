#include "config.h"

#include "AssetLoader.hpp"
#include "D3d12Application.hpp"
#include "PVR.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <array>

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

    RenderGraph::RenderPass renderPass;
    RenderGraph::RenderPipeline renderPipeline;
    renderPass.frame_buffer.color_clear_value = { 0.2f, 0.3f, 0.4f };
    renderPass.frame_buffer.color_attachments.resize(1);
    renderPass.frame_buffer.color_attachments[0].format = RenderGraph::TextureFormat::R8G8B8A8_UNORM;
    renderPass.frame_buffer.color_attachments[0].width = config.screenWidth;
    renderPass.frame_buffer.color_attachments[0].height = config.screenHeight;
    renderPass.frame_buffer.color_attachments[0].scale_x = 1.0f;
    renderPass.frame_buffer.color_attachments[0].scale_y = 1.0f;
    renderPass.frame_buffer.color_attachments[0].load_action = RenderGraph::RenderTargetLoadStoreAction::Clear;
    renderPass.frame_buffer.color_attachments[0].store_action = RenderGraph::RenderTargetLoadStoreAction::Keep;

    renderPass.frame_buffer.depth_clear_value = 0.0f;
    renderPass.frame_buffer.depth_attachment.format = RenderGraph::TextureFormat::D32_FLOAT;
    renderPass.frame_buffer.depth_attachment.width = config.screenWidth;
    renderPass.frame_buffer.depth_attachment.height = config.screenHeight;
    renderPass.frame_buffer.depth_attachment.scale_x = 1.0f;
    renderPass.frame_buffer.depth_attachment.scale_y = 1.0f;
    renderPass.frame_buffer.depth_attachment.load_action = RenderGraph::RenderTargetLoadStoreAction::Clear;
    renderPass.frame_buffer.depth_attachment.store_action = RenderGraph::RenderTargetLoadStoreAction::DontCare;

    renderPass.pipeline_state.topology_type = RenderGraph::TopologyType::Triangle;
    renderPass.pipeline_state.blend_state.enable = false;
    renderPass.pipeline_state.blend_state.separate_blend = false;
    renderPass.pipeline_state.blend_state.render_target_blend0.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend0.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend0.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend0.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend0.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend0.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend0.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend0.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend1.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend1.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend1.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend1.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend1.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend1.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend1.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend1.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend2.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend2.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend2.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend2.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend2.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend2.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend2.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend2.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend3.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend3.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend3.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend3.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend3.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend3.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend3.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend3.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend4.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend4.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend4.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend4.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend4.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend4.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend4.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend4.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend5.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend5.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend5.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend5.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend5.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend5.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend5.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend5.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend6.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend6.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend6.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend6.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend6.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend6.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend6.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend6.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.blend_state.render_target_blend7.blend_enable = false;
    renderPass.pipeline_state.blend_state.render_target_blend7.blend_operation = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend7.blend_operation_alpha = RenderGraph::BlendOperation::Add;
    renderPass.pipeline_state.blend_state.render_target_blend7.color_write_mask = 0x0F;
    renderPass.pipeline_state.blend_state.render_target_blend7.src_blend = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend7.src_blend_alpha = RenderGraph::Blend::One;
    renderPass.pipeline_state.blend_state.render_target_blend7.dst_blend = RenderGraph::Blend::Zero;
    renderPass.pipeline_state.blend_state.render_target_blend7.dst_blend_alpha = RenderGraph::Blend::Zero;

    renderPass.pipeline_state.depth_stencil_state.enable = true;
    renderPass.pipeline_state.depth_stencil_state.back_face.depth_fail = RenderGraph::StencilOperation::Keep;
    renderPass.pipeline_state.depth_stencil_state.back_face.fail = RenderGraph::StencilOperation::Keep;
    renderPass.pipeline_state.depth_stencil_state.back_face.pass = RenderGraph::StencilOperation::Keep;
    renderPass.pipeline_state.depth_stencil_state.back_face.func = RenderGraph::ComparisonFunction::Always;
    renderPass.pipeline_state.depth_stencil_state.front_face.depth_fail = RenderGraph::StencilOperation::Keep;
    renderPass.pipeline_state.depth_stencil_state.front_face.fail = RenderGraph::StencilOperation::Keep;
    renderPass.pipeline_state.depth_stencil_state.front_face.pass = RenderGraph::StencilOperation::Keep;
    renderPass.pipeline_state.depth_stencil_state.front_face.func = RenderGraph::ComparisonFunction::Always;
    renderPass.pipeline_state.depth_stencil_state.depth_function = RenderGraph::ComparisonFunction::LessEqual;
    renderPass.pipeline_state.depth_stencil_state.depth_write_mask = RenderGraph::DepthWriteMask::All;
    renderPass.pipeline_state.depth_stencil_state.stencil_enable = false;
    renderPass.pipeline_state.depth_stencil_state.stencil_read_mask = 0xFF;
    renderPass.pipeline_state.depth_stencil_state.stencil_write_mask = 0xFF;

    renderPass.pipeline_state.rasterizer_state.conservative = false;
    renderPass.pipeline_state.rasterizer_state.cull_mode = RenderGraph::CullMode::Back;
    renderPass.pipeline_state.rasterizer_state.depth_bias = 0;
    renderPass.pipeline_state.rasterizer_state.depth_bias_clamp = 0;
    renderPass.pipeline_state.rasterizer_state.depth_clip_enabled = false;
    renderPass.pipeline_state.rasterizer_state.fill_mode = RenderGraph::FillMode::Solid;
    renderPass.pipeline_state.rasterizer_state.front_counter_clockwise = true;
    renderPass.pipeline_state.rasterizer_state.multisample_enabled = true;
    renderPass.pipeline_state.rasterizer_state.slope_scaled_depth_bias = 0;

    renderPipeline.render_passes.emplace_back(renderPass);

    // 创建并登记资源创建回调函数
    ID3D12PipelineState* pPipelineState;
    ID3D12RootSignature* pRootSignature;
    ID3D12Resource* pTexture;
    std::vector<D3d12RHI::IndexBuffer> IndexBuffers;
    std::vector<D3d12RHI::VertexBuffer> VertexBuffers;
    std::array<D3d12RHI::ConstantBuffer, GfxConfiguration::kMaxInFlightFrameCount> ConstantBuffers;
    ID3D12DescriptorHeap* pCbvSrvUavHeap;
    ID3D12DescriptorHeap* pSamplerHeap;

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
        D3D12_RASTERIZER_DESC rsd = rhi.getRasterizerDesc(renderPass.pipeline_state.rasterizer_state);

        const D3D12_BLEND_DESC bld{ renderPass.pipeline_state.blend_state.enable,
                                    renderPass.pipeline_state.blend_state.separate_blend,
                                    {
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend0),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend1),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend2),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend3),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend4),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend5),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend6),
                                        rhi.getRenderTargetBlendDesc(renderPass.pipeline_state.blend_state.render_target_blend7)
                                    }};


        D3D12_DEPTH_STENCIL_DESC dsd{ renderPass.pipeline_state.depth_stencil_state.enable,
                                        rhi.getDepthWriteMask(renderPass.pipeline_state.depth_stencil_state.depth_write_mask),
                                        rhi.getCompareFunc(renderPass.pipeline_state.depth_stencil_state.depth_function),
                                        renderPass.pipeline_state.depth_stencil_state.stencil_enable,
                                        (UINT8)renderPass.pipeline_state.depth_stencil_state.stencil_read_mask,
                                        (UINT8)renderPass.pipeline_state.depth_stencil_state.stencil_write_mask,
                                        rhi.getDepthStencilOpDesc(renderPass.pipeline_state.depth_stencil_state.front_face),
                                        rhi.getDepthStencilOpDesc(renderPass.pipeline_state.depth_stencil_state.back_face)};

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

        psod.PrimitiveTopologyType = rhi.getTopologyType(renderPass.pipeline_state.topology_type);
        psod.NumRenderTargets = renderPass.frame_buffer.color_attachments.size();
        psod.SampleDesc.Count = config.msaaSamples;
        if (config.msaaSamples > 1) {
            psod.SampleDesc.Quality =
                DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
        } else {
            psod.SampleDesc.Quality = 0;
        }
        psod.DSVFormat = rhi.getDxgiFormat(renderPass.frame_buffer.depth_attachment.format);
        psod.RTVFormats[0] = rhi.getDxgiFormat(renderPass.frame_buffer.color_attachments[0].format);

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
    pSamplerHeap = rhi.CreateTextureSampler(8);

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
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        ConstantBuffers[i].size = ALIGN(sizeof(PerFrameConstants), 256);
        ConstantBuffers[i].buffer = rhi.CreateUniformBuffers(ConstantBuffers[i].size, L"Per Frame Constant Buffer");
    }

    // 创建资源描述子池
    pCbvSrvUavHeap = rhi.CreateDescriptorHeap(3, L"CbvSrvUav Heap");

    // 创建并登记资源销毁回调函数
    D3d12RHI::DestroyResourceFunc destroyResourceFunc =
        [&]() {
            SafeRelease(&pPipelineState);
            SafeRelease(&pRootSignature);

            SafeRelease(&pTexture);

            for (auto& buf : VertexBuffers) {
                SafeRelease(&buf.buffer);
            }

            for (auto& buf : IndexBuffers) {
                SafeRelease(&buf.buffer);
            }

            for (auto& buf : ConstantBuffers) {
                SafeRelease(&buf.buffer);
            }

            SafeRelease(&pCbvSrvUavHeap);
            SafeRelease(&pSamplerHeap);
        };

    rhi.DestroyResourceCB(destroyResourceFunc);

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

        rhi.UpdateUniformBufer(ConstantBuffers.data(), &ubo);

        // 绘制一帧
        rhi.BeginFrame();
        rhi.BeginPass(renderPass.frame_buffer.color_clear_value);
        rhi.SetPipelineState(pPipelineState);
        rhi.SetRootSignature(pRootSignature);
        std::array<D3d12RHI::ConstantBuffer*, 1> constBuffers = {ConstantBuffers.data()};
        rhi.CreateDescriptorSet(pCbvSrvUavHeap, 0, constBuffers.data(), 1);
        rhi.CreateDescriptorSet(pCbvSrvUavHeap, 1, &pTexture, 1);
        rhi.Draw(VertexBuffers[0].descriptor, IndexBuffers[0].descriptor,
                 pCbvSrvUavHeap, pSamplerHeap,
                 D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                 IndexBuffers[0].indexCount);
        rhi.EndPass();
        rhi.EndFrame();

        rhi.Present();
    }

    app.Finalize();

    return 0;
}