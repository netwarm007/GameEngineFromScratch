#include "D3d12GraphicsManager.hpp"

#include <objbase.h>

#include <iostream>

#include "AssetLoader.hpp"
#include "D3d12Application.hpp"
#include "D3d12Utility.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"

#include "D3d12RHI.hpp"
#include "D3d12ShadowUtils.hpp"

#include "imgui_impl_dx12.h"
#include "imgui_impl_win32.h"

#include "cbuffer.h"

#ifdef DEBUG
#define my_assert(x) assert(x)
#else
#define my_assert(x) if(!(x)) {std::cerr << "Assert failed\n"; exit(-1); }
#endif

using namespace My;
using namespace std;

namespace {
constexpr uint32_t kDescriptorsPerDraw = 14;
constexpr uint32_t kMaxDrawDescriptors =
    GfxConfiguration::kMaxSceneObjectCount *
    (2 * GfxConfiguration::kMaxShadowMapCount +
     GfxConfiguration::kMaxCubeShadowMapCount + 1) * kDescriptorsPerDraw;
}

D3d12GraphicsManager::~D3d12GraphicsManager() {
    for (auto& buf : m_VertexBuffers) {
        SafeRelease(&buf.buffer);
    }

    for (auto& buf : m_IndexBuffers) {
        SafeRelease(&buf.buffer);
    }

    for (auto& buf : m_PerFrameConstantBuffers) {
        SafeRelease(&buf.buffer);
    }

    for (auto& buf : m_PerBatchConstantBuffers) {
        SafeRelease(&buf.buffer);
    }
}

int D3d12GraphicsManager::Initialize() {
    int result = GraphicsManager::Initialize();

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto pDev = rhi.GetDevice();

    D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc{};
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvSrvUavHeapDesc.NumDescriptors = 4096;

    my_assert(SUCCEEDED(pDev->CreateDescriptorHeap(
        &cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_pCbvSrvUavHeapImGui))));
    m_pCbvSrvUavHeapImGui->SetName(L"ImGui CbvSrvUav Heap");

    // ImGui allocates its own texture descriptors (e.g. the font atlas) from
    // the same slot pool used by CreateImGuiSrv
    ImGui_ImplDX12_InitInfo imguiInitInfo;
    imguiInitInfo.Device = pDev;
    imguiInitInfo.CommandQueue = rhi.GetGraphicsCommandQueue();
    imguiInitInfo.NumFramesInFlight = GfxConfiguration::kMaxInFlightFrameCount;
    imguiInitInfo.RTVFormat = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    imguiInitInfo.DSVFormat = ::DXGI_FORMAT_UNKNOWN;
    imguiInitInfo.UserData = this;
    imguiInitInfo.SrvDescriptorHeap = m_pCbvSrvUavHeapImGui;
    imguiInitInfo.SrvDescriptorAllocFn =
        [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu,
           D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu) {
            auto* self = static_cast<D3d12GraphicsManager*>(info->UserData);
            auto& rhi = dynamic_cast<D3d12Application*>(self->m_pApp)->GetRHI();
            uint32_t slot;
            if (!self->m_freeImGuiSlots.empty()) {
                slot = self->m_freeImGuiSlots.back();
                self->m_freeImGuiSlots.pop_back();
            } else {
                my_assert(self->m_nextImGuiSlot < 4096);
                slot = self->m_nextImGuiSlot++;
            }
            *out_cpu = rhi.GetCpuDescriptorHandle(self->m_pCbvSrvUavHeapImGui, slot);
            *out_gpu = rhi.GetGpuDescriptorHandle(self->m_pCbvSrvUavHeapImGui, slot);
        };
    imguiInitInfo.SrvDescriptorFreeFn =
        [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
           D3D12_GPU_DESCRIPTOR_HANDLE) {
            auto* self = static_cast<D3d12GraphicsManager*>(info->UserData);
            auto& rhi = dynamic_cast<D3d12Application*>(self->m_pApp)->GetRHI();
            const auto base = rhi.GetCpuDescriptorHandle(self->m_pCbvSrvUavHeapImGui, 0);
            const auto stride = info->Device->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            self->m_freeImGuiSlots.push_back(
                static_cast<uint32_t>((cpu.ptr - base.ptr) / stride));
        };
    ImGui_ImplDX12_Init(&imguiInitInfo);

    // 创建常量缓冲区
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        m_PerFrameConstantBuffers[i].size = sizeof(PerFrameConstants);
        m_PerFrameConstantBuffers[i].buffer = rhi.CreateUniformBuffers(m_PerFrameConstantBuffers[i].size, L"Per Frame Constant Buffer");
        m_PerBatchConstantBuffers[i].size = sizeof(PerBatchConstants) * GfxConfiguration::kMaxSceneObjectCount;
        m_PerBatchConstantBuffers[i].buffer = rhi.CreateUniformBuffers(m_PerBatchConstantBuffers[i].size, L"Per Batch Constant Buffer");
        m_LightBuffers[i].size = sizeof(LightInfo);
        m_LightBuffers[i].buffer = rhi.CreateUniformBuffers(sizeof(LightInfo), L"Light Buffer");
        m_ShadowBuffers[i].size = kSizeShadowMapConstantBuffer * MAX_LIGHTS;
        m_ShadowBuffers[i].buffer = rhi.CreateUniformBuffers(m_ShadowBuffers[i].size, L"Shadow Constants");
        m_pDrawHeaps[i] = rhi.CreateDescriptorHeap(kMaxDrawDescriptors, L"Draw Descriptor Heap");
    }

    D3D12_DESCRIPTOR_HEAP_DESC shadowDsvDesc{};
    shadowDsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    shadowDsvDesc.NumDescriptors = 1;
    my_assert(SUCCEEDED(pDev->CreateDescriptorHeap(&shadowDsvDesc, IID_PPV_ARGS(&m_pShadowDsvHeap))));

    // 创建资源描述子池
    m_pCbvSrvUavHeap = rhi.CreateDescriptorHeap(2, L"CbvSrvUav Heap");
    m_pComputeHeap = rhi.CreateDescriptorHeap(1, L"Compute UAV Heap");

    // 创建采样器
    m_pSamplerHeap = rhi.CreateTextureSampler(8);

    return result;
}

void D3d12GraphicsManager::Finalize() {
    dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().WaitForIdle();
    m_DrawPasses.clear();
    GraphicsManager::Finalize();
    ImGui_ImplDX12_Shutdown();

    SafeRelease(&m_pCbvSrvUavHeapImGui);
    SafeRelease(&m_pCbvSrvUavHeap);
    SafeRelease(&m_pComputeHeap);
    SafeRelease(&m_pMipmapPipeline);
    SafeRelease(&m_pMipmapRoot);
    SafeRelease(&m_pSamplerHeap);
    SafeRelease(&m_pShadowDsvHeap);
    for (auto& heap : m_pDrawHeaps) SafeRelease(&heap);
    for (auto& buffer : m_LightBuffers) SafeRelease(&buffer.buffer);
    for (auto& buffer : m_ShadowBuffers) SafeRelease(&buffer.buffer);
    for (auto& buffer : m_PerFrameConstantBuffers) SafeRelease(&buffer.buffer);
    for (auto& buffer : m_PerBatchConstantBuffers) SafeRelease(&buffer.buffer);
}

void D3d12GraphicsManager::Tick() {
    m_nFrameIndex = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().GetCurrentFrameIndex();
    GraphicsManager::Tick();
}

void D3d12GraphicsManager::ResizeCanvas(int32_t width, int32_t height) {
    if (width > 0 && height > 0 && (m_canvasWidth != width || m_canvasHeight != height)) {
        m_pendingWidth = width;
        m_pendingHeight = height;
    }
}

size_t D3d12GraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
    const void* pData = v_property_array.GetData();
    auto element_size = v_property_array.GetVertexCount();
    auto stride = v_property_array.GetDataSize() / element_size;

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    m_VertexBuffers.emplace_back(rhi.CreateVertexBuffer(pData, element_size, (int32_t)stride));

    return m_VertexBuffers.size() - 1;
}

size_t D3d12GraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
    const void* pData = index_array.GetData();
    auto element_size = index_array.GetIndexCount();
    int32_t index_size =
        static_cast<int32_t>(index_array.GetDataSize() / element_size);

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    m_IndexBuffers.emplace_back(rhi.CreateIndexBuffer(pData, element_size, index_size));

    return m_IndexBuffers.size() - 1;
}

// this is the function that loads and prepares the pso
HRESULT D3d12GraphicsManager::CreatePSO(D3d12PipelineState& pipelineState) {
    HRESULT hr = S_OK;

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    if (pipelineState.pipelineType == PIPELINE_TYPE::GRAPHIC) {
        D3D12_SHADER_BYTECODE vertexShaderByteCode;
        vertexShaderByteCode.pShaderBytecode =
            pipelineState.vertexShaderByteCode.pShaderBytecode;
        vertexShaderByteCode.BytecodeLength =
            pipelineState.vertexShaderByteCode.BytecodeLength;

        D3D12_SHADER_BYTECODE pixelShaderByteCode;
        pixelShaderByteCode.pShaderBytecode =
            pipelineState.pixelShaderByteCode.pShaderBytecode;
        pixelShaderByteCode.BytecodeLength =
            pipelineState.pixelShaderByteCode.BytecodeLength;

        // create rasterizer descriptor
        D3D12_RASTERIZER_DESC rsd{D3D12_FILL_MODE_SOLID,
                                  D3D12_CULL_MODE_BACK,
                                  TRUE,
                                  D3D12_DEFAULT_DEPTH_BIAS,
                                  D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                                  TRUE,
                                  FALSE,
                                  FALSE,
                                  0,
                                  D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF};

        switch (pipelineState.cullFaceMode) {
            case CULL_FACE_MODE::FRONT:
                rsd.CullMode = D3D12_CULL_MODE_FRONT;
                break;
            case CULL_FACE_MODE::BACK:
                rsd.CullMode = D3D12_CULL_MODE_BACK;
                break;
            case CULL_FACE_MODE::NONE:
                rsd.CullMode = D3D12_CULL_MODE_NONE;
                break;
            default:
                my_assert(0);
        }
        if (pipelineState.geometryShaderName == "shadowmap_omni.geom") {
            // The point-shadow projection flips Y, reversing triangle winding.
            rsd.FrontCounterClockwise = FALSE;
        }

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
            D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
            D3D12_COMPARISON_FUNC_ALWAYS};

        D3D12_DEPTH_STENCIL_DESC dsd{TRUE,
                                     D3D12_DEPTH_WRITE_MASK_ALL,
                                     D3D12_COMPARISON_FUNC_LESS,
                                     FALSE,
                                     D3D12_DEFAULT_STENCIL_READ_MASK,
                                     D3D12_DEFAULT_STENCIL_WRITE_MASK,
                                     defaultStencilOp,
                                     defaultStencilOp};

        if (pipelineState.bDepthWrite) {
            dsd.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        } else {
            dsd.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        }

        if (pipelineState.depthTestMode == DEPTH_TEST_MODE::NONE) {
            dsd.DepthEnable = FALSE;
        } else {
            dsd.DepthEnable = TRUE;
            switch (pipelineState.depthTestMode) {
                case DEPTH_TEST_MODE::ALWAYS:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
                    break;
                case DEPTH_TEST_MODE::EQUAL:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
                    break;
                case DEPTH_TEST_MODE::LARGE:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
                    break;
                case DEPTH_TEST_MODE::LARGE_EQUAL:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
                    break;
                case DEPTH_TEST_MODE::LESS:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
                    break;
                case DEPTH_TEST_MODE::LESS_EQUAL:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                    break;
                case DEPTH_TEST_MODE::NEVER:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_NEVER;
                    break;
                default:
                    my_assert(0);
            }
        }

        // create the root signature
        pipelineState.rootSignature =
            rhi.CreateRootSignature(pixelShaderByteCode);

        // create the input layout object
        static const D3D12_INPUT_ELEMENT_DESC ied_full[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 1, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TANGENT", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 3, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        static const D3D12_INPUT_ELEMENT_DESC ied_simple[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        static const D3D12_INPUT_ELEMENT_DESC ied_cube[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 3, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        static const D3D12_INPUT_ELEMENT_DESC ied_pos_only[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        // describe and create the graphics pipeline state object (PSO)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psod{};
        psod.pRootSignature = pipelineState.rootSignature;
        psod.VS = vertexShaderByteCode;
        psod.PS = pixelShaderByteCode;
        psod.GS = pipelineState.geometryShaderByteCode;
        psod.BlendState = bld;
        psod.SampleMask = UINT_MAX;
        psod.RasterizerState = rsd;
        psod.DepthStencilState = dsd;
        switch (pipelineState.a2vType) {
            case A2V_TYPES::A2V_TYPES_FULL:
                psod.InputLayout = {ied_full, _countof(ied_full)};
                break;
            case A2V_TYPES::A2V_TYPES_SIMPLE:
                psod.InputLayout = {ied_simple, _countof(ied_simple)};
                break;
            case A2V_TYPES::A2V_TYPES_CUBE:
                psod.InputLayout = {ied_cube, _countof(ied_cube)};
                break;
            case A2V_TYPES::A2V_TYPES_POS_ONLY:
                psod.InputLayout = {ied_pos_only, _countof(ied_pos_only)};
                break;
            default:
                my_assert(0);
        }

        psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        if (pipelineState.flag == PIPELINE_FLAG::SHADOW) {
            psod.NumRenderTargets = 0;
            psod.RTVFormats[0] = ::DXGI_FORMAT_UNKNOWN;
            psod.SampleDesc.Count = 1;
            psod.SampleDesc.Quality = 0;
        } else {
            psod.NumRenderTargets = 1;
            psod.RTVFormats[0] = ::DXGI_FORMAT_R8G8B8A8_UNORM;
            psod.SampleDesc.Count = m_pApp->GetConfiguration().msaaSamples;
            psod.SampleDesc.Quality = psod.SampleDesc.Count > 1
                ? DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN : 0;
        }
        psod.DSVFormat = ::DXGI_FORMAT_D32_FLOAT;

        pipelineState.pipelineState = rhi.CreateGraphicsPipeline(psod);
    } else {
        my_assert(pipelineState.pipelineType == PIPELINE_TYPE::COMPUTE);

        D3D12_SHADER_BYTECODE computeShaderByteCode;
        computeShaderByteCode.pShaderBytecode =
            pipelineState.computeShaderByteCode.pShaderBytecode;
        computeShaderByteCode.BytecodeLength =
            pipelineState.computeShaderByteCode.BytecodeLength;

        // create the root signature
        pipelineState.rootSignature =
            rhi.CreateRootSignature(computeShaderByteCode);

        D3D12_CACHED_PIPELINE_STATE cachedPSO;
        cachedPSO.pCachedBlob = nullptr;
        cachedPSO.CachedBlobSizeInBytes = 0;
        D3D12_COMPUTE_PIPELINE_STATE_DESC psod;
        psod.pRootSignature = pipelineState.rootSignature;
        psod.CS = computeShaderByteCode;
        psod.NodeMask = 0;
        psod.CachedPSO.pCachedBlob = nullptr;
        psod.CachedPSO.CachedBlobSizeInBytes = 0;
        psod.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        pipelineState.pipelineState = rhi.CreateComputePipeline(psod);
    }

    pipelineState.pipelineState->SetName(
        s2ws(pipelineState.pipelineStateName).c_str());

    return hr;
}

void D3d12GraphicsManager::initializeGeometries(const Scene& scene) {
    cout << "Creating Draw Batch Contexts ...";
    uint32_t batch_index = 0;
    for (const auto& _it : scene.GeometryNodes) {
        const auto& pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible()) {
            const auto& pGeometry =
                scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            my_assert(pGeometry);
            const auto& pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            // Set the number of vertex properties.
            const auto vertexPropertiesCount =
                pMesh->GetVertexPropertiesCount();

            const auto propertyOffset = m_VertexBuffers.size();
            for (uint32_t i = 0; i < vertexPropertiesCount; i++) {
                const SceneObjectVertexArray& v_property_array =
                    pMesh->GetVertexPropertyArray(i);
                CreateVertexBuffer(v_property_array);
            }

            for (size_t group = 0; group < pMesh->GetIndexGroupCount(); ++group) {
                auto dbc = make_shared<D3dDrawBatchContext>();
                dbc->property_offset = propertyOffset;
                const SceneObjectIndexArray& index_array = pMesh->GetIndexArray(group);
                dbc->index_offset = CreateIndexBuffer(index_array);

                const auto material_index = index_array.GetMaterialIndex();
                const auto material_key =
                    pGeometryNode->GetMaterialRef(material_index);
                const auto& material = scene.GetMaterial(material_key);

                dbc->batchIndex = batch_index++;
                dbc->index_count = (UINT)index_array.GetIndexCount();
                dbc->property_count = vertexPropertiesCount;

                if (material) {
                    auto loadTexture = [this](const auto& texture, Texture2D& target) {
                        if (!texture) return;
                        CreateTexture(*texture);
                        auto it = m_materialTextures.find(texture->GetName());
                        if (it != m_materialTextures.end()) target = it->second;
                    };
                    loadTexture(material->GetBaseColor().ValueMap, dbc->material.diffuseMap);
                    loadTexture(material->GetNormal().ValueMap, dbc->material.normalMap);
                    loadTexture(material->GetMetallic().ValueMap, dbc->material.metallicMap);
                    loadTexture(material->GetRoughness().ValueMap, dbc->material.roughnessMap);
                    loadTexture(material->GetAO().ValueMap, dbc->material.aoMap);
                    loadTexture(material->GetHeight().ValueMap, dbc->material.heightMap);
                }

                dbc->node = pGeometryNode;

                for (auto& frame : m_Frames) {
                    frame.batchContexts.push_back(dbc);
                }
            }
        }
    }
    cout << "Done!" << endl;
}

void D3d12GraphicsManager::initializeSkyBox(const Scene& scene) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    my_assert(scene.SkyBox);

    m_dbcSkyBox.property_offset = m_VertexBuffers.size();
    m_VertexBuffers.emplace_back(rhi.CreateVertexBuffer(SceneObjectSkyBox::skyboxVertices.data(),
                            SceneObjectSkyBox::skyboxVertices.size() / 3,
                            3 * sizeof(SceneObjectSkyBox::skyboxVertices[0])));
    m_dbcSkyBox.property_count = 1;
    m_dbcSkyBox.index_offset = m_IndexBuffers.size();
    m_IndexBuffers.emplace_back(rhi.CreateIndexBuffer(
        SceneObjectSkyBox::skyboxIndices.data(),
        SceneObjectSkyBox::skyboxIndices.size(),
        static_cast<int32_t>(sizeof(SceneObjectSkyBox::skyboxIndices[0]))));
    m_dbcSkyBox.index_count = SceneObjectSkyBox::skyboxIndices.size();
    BuildIdentityMatrix(m_dbcSkyBox.modelMatrix);

    auto image = scene.SkyBox->GetTexture(0).GetTextureImage();
    auto radiance = scene.SkyBox->GetTexture(12).GetTextureImage();
    my_assert(image && radiance && !image->mipmaps.empty() && !radiance->mipmaps.empty());
    const auto mipLevels = static_cast<uint16_t>(std::max<size_t>(2, radiance->mipmaps.size()));
    const auto format = rhi.getDxgiFormat(*image);
    auto* texture = rhi.CreateTextureCubeArray(format, image->Width, image->Height, 2, mipLevels);
    for (uint32_t i = 0; i < 18; ++i) {
        auto face = scene.SkyBox->GetTexture(i).GetTextureImage();
        my_assert(face && rhi.getDxgiFormat(*face) == format && !face->mipmaps.empty());
        const uint32_t slice = i < 12 ? i % 6 : i % 6 + 6;
        if (i < 12) {
            const uint32_t mip = i / 6;
            my_assert(face->Width == (std::max)(1u, image->Width >> mip) &&
                      face->Height == (std::max)(1u, image->Height >> mip));
            rhi.UploadTextureSubresource(texture, *face, 0, mip + slice * mipLevels);
        } else {
            my_assert(face->Width == image->Width && face->Height == image->Height);
            for (uint32_t mip = 0; mip < face->mipmaps.size() && mip < mipLevels; ++mip) {
                rhi.UploadTextureSubresource(texture, *face, mip, mip + slice * mipLevels);
            }
        }
    }
    auto& skybox = m_Frames[0].skybox;
    skybox.buff = reinterpret_cast<TextureHandler>(texture);
    skybox.handler = skybox.buff;
    skybox.format = format;
    skybox.pixel_format = image->pixel_format;
    skybox.width = image->Width;
    skybox.height = image->Height;
    skybox.mips = mipLevels;
    skybox.size = 2;
    m_Textures.push_back(skybox);
}

void D3d12GraphicsManager::EndScene() {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    rhi.WaitForIdle();
    rhi.ResetAllBuffers();
    for (auto& buffer : m_VertexBuffers) SafeRelease(&buffer.buffer);
    for (auto& buffer : m_IndexBuffers) SafeRelease(&buffer.buffer);
    m_VertexBuffers.clear();
    m_IndexBuffers.clear();

    std::unordered_set<TextureHandler> released;
    for (auto& texture : m_Textures) {
        if (texture.buff && released.insert(texture.buff).second) ReleaseTexture(texture);
    }
    for (auto& frame : m_Frames) {
        for (auto& texture : frame.colorTextures) {
            if (texture.buff && released.insert(texture.buff).second) ReleaseTexture(texture);
        }
        if (frame.depthTexture.buff && released.insert(frame.depthTexture.buff).second) {
            ReleaseTexture(frame.depthTexture);
        }
        frame.colorTextures.clear();
        frame.depthTexture = {};
        frame.batchContexts.clear();
    }
    m_Textures.clear();
    m_materialTextures.clear();
    m_RenderedShadowMaps.clear();
    for (auto& frame : m_Frames) {
        frame.brdfLUT = {};
        frame.skybox = {};
        frame.frameContext.shadowMap = {};
        frame.frameContext.globalShadowMap = {};
        frame.frameContext.cubeShadowMap = {};
    }
}

void D3d12GraphicsManager::BeginFrame(Frame& frame) {
    GraphicsManager::BeginFrame(frame);
    ImGui_ImplDX12_NewFrame();

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    if (m_pendingWidth > 0 && m_pendingHeight > 0) {
        rhi.WaitForIdle();
        GraphicsManager::ResizeCanvas(m_pendingWidth, m_pendingHeight);
        m_pendingWidth = m_pendingHeight = 0;
    }
    rhi.BeginFrame();
    m_nextDrawDescriptor = 0;

    my_assert(frame.frameIndex == m_nFrameIndex);
}

void D3d12GraphicsManager::EndFrame(Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    UpdateDebugViews();
    rhi.EndFrame(false);
    rhi.DrawGUI(m_pCbvSrvUavHeapImGui);
    GraphicsManager::EndFrame(frame);
}

void D3d12GraphicsManager::BeginPass(Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    if (!frame.renderToTexture) return;
    auto& color = frame.colorTextures[frame.enableMSAA ? 1 : 0];
    auto* colorResource = reinterpret_cast<ID3D12Resource*>(color.buff);
    auto* depthResource = reinterpret_cast<ID3D12Resource*>(frame.depthTexture.buff);
    TransitionTexture(colorResource, D3D12_RESOURCE_STATE_RENDER_TARGET);
    auto* cmd = rhi.GetGraphicsCommandList();
    const auto rtv = m_renderTextures.at(colorResource).viewHeap->GetCPUDescriptorHandleForHeapStart();
    const auto dsv = m_renderTextures.at(depthResource).viewHeap->GetCPUDescriptorHandleForHeapStart();
    cmd->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    D3D12_VIEWPORT viewport{0, 0, static_cast<float>(color.width), static_cast<float>(color.height), 0, 1};
    D3D12_RECT scissor{0, 0, static_cast<LONG>(color.width), static_cast<LONG>(color.height)};
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);
    if (frame.clearRT) {
        const float clear[] = {frame.clearColor[0], frame.clearColor[1], frame.clearColor[2], 1.0f};
        cmd->ClearRenderTargetView(rtv, clear, 0, nullptr);
        cmd->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    }
}

void D3d12GraphicsManager::EndPass(Frame& frame) {
    if (frame.renderToTexture && !frame.enableMSAA) {
        TransitionTexture(reinterpret_cast<ID3D12Resource*>(frame.colorTextures[0].buff),
                          D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
}

void D3d12GraphicsManager::DrawBatch(const Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    for (const auto& pDbc : frame.batchContexts) {
        const D3dDrawBatchContext& dbc =
            dynamic_cast<const D3dDrawBatchContext&>(*pDbc);
        auto* cmd = rhi.GetGraphicsCommandList();
        std::vector<D3D12_VERTEX_BUFFER_VIEW> views;
        views.reserve(dbc.property_count);
        for (size_t i = 0; i < dbc.property_count; ++i) {
            views.push_back(m_VertexBuffers[dbc.property_offset + i].descriptor);
        }
        cmd->IASetVertexBuffers(0, static_cast<UINT>(views.size()), views.data());
        cmd->IASetIndexBuffer(&m_IndexBuffers[dbc.index_offset].descriptor);
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        BindDrawResources(frame, dbc);
        cmd->DrawIndexedInstanced(dbc.index_count, 1, 0, 0, 0);
    }
}

void D3d12GraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    if (pipelineState) {
        std::shared_ptr<D3d12PipelineState> pState =
            dynamic_pointer_cast<D3d12PipelineState>(pipelineState);

        if (!pState->pipelineState) {
            CreatePSO(*pState);
        }

        if (pState->pipelineType == PIPELINE_TYPE::COMPUTE) {
            rhi.SetComputePipelineState(pState->pipelineState, pState->rootSignature);
        } else {
            rhi.SetRootSignature(pState->rootSignature);
            rhi.SetPipelineState(pState->pipelineState);
            SetPerFrameConstants(frame);
            SetLightInfo(frame);
        }
    }
}

void D3d12GraphicsManager::SetPerFrameConstants(const Frame& frame) {
    auto& buffer = m_PerFrameConstantBuffers[frame.frameIndex];
    void* data = nullptr;
    D3D12_RANGE readRange{0, 0};
    my_assert(SUCCEEDED(buffer.buffer->Map(0, &readRange, &data)));
    memcpy(data, static_cast<const PerFrameConstants*>(&frame.frameContext), sizeof(PerFrameConstants));
    buffer.buffer->Unmap(0, nullptr);
}

void D3d12GraphicsManager::SetLightInfo(const Frame& frame) {
    auto& buffer = m_LightBuffers[frame.frameIndex];
    void* data = nullptr;
    D3D12_RANGE readRange{0, 0};
    my_assert(SUCCEEDED(buffer.buffer->Map(0, &readRange, &data)));
    memcpy(data, &frame.lightInfo, sizeof(frame.lightInfo));
    buffer.buffer->Unmap(0, nullptr);
}

void D3d12GraphicsManager::BindDrawResources(const Frame& frame, const DrawBatchContext& batch) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* device = rhi.GetDevice();
    auto* cmd = rhi.GetGraphicsCommandList();
    my_assert(m_nextDrawDescriptor + kDescriptorsPerDraw <= kMaxDrawDescriptors);
    auto* heap = m_pDrawHeaps[frame.frameIndex];
    auto cpu = rhi.GetCpuDescriptorHandle(heap, m_nextDrawDescriptor);
    auto gpu = rhi.GetGpuDescriptorHandle(heap, m_nextDrawDescriptor);
    const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{};
    cbv.BufferLocation = m_LightBuffers[frame.frameIndex].buffer->GetGPUVirtualAddress();
    cbv.SizeInBytes = static_cast<UINT>(kSizeLightInfo);
    device->CreateConstantBufferView(&cbv, cpu);
    cpu.ptr += increment;
    cbv.BufferLocation = m_ShadowBuffers[frame.frameIndex].buffer->GetGPUVirtualAddress();
    if (m_currentShadowLight >= 0) cbv.BufferLocation += m_currentShadowLight * kSizeShadowMapConstantBuffer;
    cbv.SizeInBytes = static_cast<UINT>(kSizeShadowMapConstantBuffer);
    device->CreateConstantBufferView(&cbv, cpu);
    cpu.ptr += increment;

    const TextureBase* textures[] = {
        &batch.material.diffuseMap, &batch.material.normalMap,
        &batch.material.metallicMap, &batch.material.roughnessMap,
        &batch.material.aoMap, &batch.material.heightMap, &frame.brdfLUT,
        &frame.frameContext.shadowMap, &frame.frameContext.globalShadowMap,
        &frame.frameContext.cubeShadowMap, &frame.skybox, &frame.terrainHeightMap
    };
    for (uint32_t i = 0; i < _countof(textures); ++i) {
        auto* resource = reinterpret_cast<ID3D12Resource*>(textures[i]->buff);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = resource ? static_cast<DXGI_FORMAT>(textures[i]->format) : DXGI_FORMAT_R8G8B8A8_UNORM;
        if (i == 7 || i == 8) {
            srv.Format = DXGI_FORMAT_R32_FLOAT;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srv.Texture2DArray.MipLevels = 1;
            srv.Texture2DArray.ArraySize = resource ? resource->GetDesc().DepthOrArraySize : 1;
        } else if (i == 9 || i == 10) {
            if (i == 9) srv.Format = DXGI_FORMAT_R32_FLOAT;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            srv.TextureCubeArray.MipLevels = resource ? resource->GetDesc().MipLevels : 1;
            srv.TextureCubeArray.NumCubes = resource ? resource->GetDesc().DepthOrArraySize / 6 : 1;
        } else {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = resource ? resource->GetDesc().MipLevels : 1;
        }
        device->CreateShaderResourceView(resource, &srv, cpu);
        cpu.ptr += increment;
    }

    ID3D12DescriptorHeap* heaps[] = {heap, m_pSamplerHeap};
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);
    cmd->SetGraphicsRootConstantBufferView(0, m_PerFrameConstantBuffers[frame.frameIndex].buffer->GetGPUVirtualAddress());
    cmd->SetGraphicsRoot32BitConstants(1, 16, &batch.modelMatrix, 0);
    cmd->SetGraphicsRootDescriptorTable(2, gpu);
    cmd->SetGraphicsRootDescriptorTable(3, m_pSamplerHeap->GetGPUDescriptorHandleForHeapStart());
    m_nextDrawDescriptor += kDescriptorsPerDraw;
}

void D3d12GraphicsManager::Present() {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.Present();
    m_nFrameIndex = rhi.GetCurrentFrameIndex();
}

void D3d12GraphicsManager::DrawSkyBox(const Frame& frame) {
    if (!frame.skybox.buff) return;
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* cmd = rhi.GetGraphicsCommandList();
    cmd->IASetVertexBuffers(0, 1, &m_VertexBuffers[m_dbcSkyBox.property_offset].descriptor);
    cmd->IASetIndexBuffer(&m_IndexBuffers[m_dbcSkyBox.index_offset].descriptor);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    BindDrawResources(frame, m_dbcSkyBox);
    cmd->DrawIndexedInstanced(m_dbcSkyBox.index_count, 1, 0, 0, 0);
}

void D3d12GraphicsManager::GenerateCubemapArray(TextureCubeArray& texture_array) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    my_assert(texture_array.pixel_format == PIXEL_FORMAT::D32);
    my_assert(texture_array.size <= UINT16_MAX / 6);
    auto* texture = rhi.CreateDepthTextureArray(
        texture_array.width, texture_array.height,
        static_cast<uint16_t>(texture_array.size * 6));
    texture_array.buff = reinterpret_cast<TextureHandler>(texture);
    texture_array.handler = texture_array.buff;
    texture_array.format = DXGI_FORMAT_R32_TYPELESS;
    m_Textures.push_back(texture_array);
}

void D3d12GraphicsManager::GenerateTextureArray(Texture2DArray& texture_array) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    my_assert(texture_array.pixel_format == PIXEL_FORMAT::D32);
    my_assert(texture_array.size <= UINT16_MAX);
    auto* texture = rhi.CreateDepthTextureArray(
        texture_array.width, texture_array.height,
        static_cast<uint16_t>(texture_array.size));
    texture_array.buff = reinterpret_cast<TextureHandler>(texture);
    texture_array.handler = texture_array.buff;
    texture_array.format = DXGI_FORMAT_R32_TYPELESS;
    m_Textures.push_back(texture_array);
}

void D3d12GraphicsManager::CreateImGuiSrv(Texture2D& texture,
                                           const D3D12_SHADER_RESOURCE_VIEW_DESC& desc) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    uint32_t slot;
    if (!m_freeImGuiSlots.empty()) {
        slot = m_freeImGuiSlots.back();
        m_freeImGuiSlots.pop_back();
    } else {
        my_assert(m_nextImGuiSlot < 4096);
        slot = m_nextImGuiSlot++;
    }
    const auto cpu = rhi.GetCpuDescriptorHandle(m_pCbvSrvUavHeapImGui, slot);
    const auto gpu = rhi.GetGpuDescriptorHandle(m_pCbvSrvUavHeapImGui, slot);
    rhi.GetDevice()->CreateShaderResourceView(
        reinterpret_cast<ID3D12Resource*>(texture.buff), &desc, cpu);
    texture.handler = static_cast<TextureHandler>(gpu.ptr);
    m_imguiSlots.emplace(texture.buff, slot);
}

void D3d12GraphicsManager::GenerateTexture(Texture2D& texture) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* device = rhi.GetDevice();
    const bool depth = texture.pixel_format == PIXEL_FORMAT::D32;
    my_assert(depth || texture.pixel_format == PIXEL_FORMAT::RGBA8);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = texture.width;
    desc.Height = texture.height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = depth ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = texture.samples;
    desc.SampleDesc.Quality = texture.samples > 1 ? DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN : 0;
    desc.Flags = depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = desc.Format;
    if (depth) clear.DepthStencil.Depth = 1.0f;
    else {
        clear.Color[0] = 0.2f;
        clear.Color[1] = 0.3f;
        clear.Color[2] = 0.4f;
        clear.Color[3] = 1.0f;
    }
    RenderTexture info;
    info.state = depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE :
        (texture.samples > 1 ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    ID3D12Resource* resource = nullptr;
    my_assert(SUCCEEDED(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, info.state, &clear, IID_PPV_ARGS(&resource))));
    D3D12_DESCRIPTOR_HEAP_DESC viewHeapDesc{};
    viewHeapDesc.Type = depth ? D3D12_DESCRIPTOR_HEAP_TYPE_DSV : D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    viewHeapDesc.NumDescriptors = 1;
    my_assert(SUCCEEDED(device->CreateDescriptorHeap(&viewHeapDesc, IID_PPV_ARGS(&info.viewHeap))));
    if (depth) {
        device->CreateDepthStencilView(resource, nullptr, info.viewHeap->GetCPUDescriptorHandleForHeapStart());
    } else {
        device->CreateRenderTargetView(resource, nullptr, info.viewHeap->GetCPUDescriptorHandleForHeapStart());
    }
    m_renderTextures.emplace(resource, info);
    texture.buff = reinterpret_cast<TextureHandler>(resource);
    texture.handler = texture.buff;
    texture.format = desc.Format;
    if (!depth && texture.samples == 1) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = desc.Format;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;
        CreateImGuiSrv(texture, srv);
    }
}

void D3d12GraphicsManager::CreateTextureView(Texture2D& texture_view,
                                              const TextureArrayBase& texture_array,
                                              uint32_t slice, uint32_t mip) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* source = reinterpret_cast<ID3D12Resource*>(texture_array.buff);
    my_assert(source);
    auto sourceDesc = source->GetDesc();
    my_assert(slice < sourceDesc.DepthOrArraySize && mip < sourceDesc.MipLevels);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc = sourceDesc;
    desc.Width = (std::max)(UINT64(1), sourceDesc.Width >> mip);
    desc.Height = (std::max)(1u, sourceDesc.Height >> mip);
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* target = nullptr;
    my_assert(SUCCEEDED(rhi.GetDevice()->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(&target))));
    m_renderTextures.emplace(target, RenderTexture{nullptr, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE});
    source->AddRef();
    m_debugViews.emplace(target, DebugTextureView{source, mip + slice * sourceDesc.MipLevels});
    texture_view.buff = reinterpret_cast<TextureHandler>(target);
    texture_view.format = desc.Format == DXGI_FORMAT_R32_TYPELESS ? DXGI_FORMAT_R32_FLOAT : desc.Format;
    texture_view.pixel_format = texture_array.pixel_format;
    texture_view.width = static_cast<uint32_t>(desc.Width);
    texture_view.height = desc.Height;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = static_cast<DXGI_FORMAT>(texture_view.format);
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    CreateImGuiSrv(texture_view, srv);
}

void D3d12GraphicsManager::UpdateDebugViews() {
    auto* cmd = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().GetGraphicsCommandList();
    for (const auto& entry : m_debugViews) {
        auto* target = entry.first;
        auto* source = entry.second.source;
        my_assert(m_RenderedShadowMaps.contains(source));
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = source;
        barrier.Transition.Subresource = entry.second.subresource;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        cmd->ResourceBarrier(1, &barrier);
        TransitionTexture(target, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = source;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = entry.second.subresource;
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = target;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmd->ResourceBarrier(1, &barrier);
        TransitionTexture(target, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
}

void D3d12GraphicsManager::TransitionTexture(ID3D12Resource* resource,
                                             D3D12_RESOURCE_STATES state) {
    auto& info = m_renderTextures.at(resource);
    if (info.state == state) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = info.state;
    barrier.Transition.StateAfter = state;
    dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().GetGraphicsCommandList()->ResourceBarrier(1, &barrier);
    info.state = state;
}

void D3d12GraphicsManager::MSAAResolve(std::optional<std::reference_wrapper<Texture2D>> target,
                                        Texture2D& source) {
    my_assert(target.has_value());
    auto* dst = reinterpret_cast<ID3D12Resource*>(target->get().buff);
    auto* src = reinterpret_cast<ID3D12Resource*>(source.buff);
    TransitionTexture(src, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    TransitionTexture(dst, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.GetGraphicsCommandList()->ResolveSubresource(dst, 0, src, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    TransitionTexture(src, D3D12_RESOURCE_STATE_RENDER_TARGET);
    TransitionTexture(dst, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

Texture2D My::D3d12GraphicsManager::CreateTexture(Image& img) {
    auto result = UploadTexture(img);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = static_cast<DXGI_FORMAT>(result.format);
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    CreateImGuiSrv(result, srv);
    return result;
}

Texture2D D3d12GraphicsManager::UploadTexture(Image& img) {
    Texture2D result;
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    result.format = rhi.getDxgiFormat(img);
    result.height = img.Height;
    result.width = img.Width;
    result.mips = 1;
    result.samples = 1;
    result.pixel_format = img.pixel_format;
    auto res = rhi.CreateTextureImage(img);
    result.buff = (TextureHandler)res;
    return result;
}

void My::D3d12GraphicsManager::UpdateTexture(Texture2D& texture, Image& img) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.UpdateTexture((ID3D12Resource*)texture.buff, img);
}

void D3d12GraphicsManager::BeginShadowMap(
    const int32_t light_index, const TextureBase* pShadowmap, const int32_t layer_index, const Frame& frame) {
    my_assert(pShadowmap && pShadowmap->buff && light_index >= 0 && light_index < MAX_LIGHTS);
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* cmd = rhi.GetGraphicsCommandList();
    auto* resource = reinterpret_cast<ID3D12Resource*>(pShadowmap->buff);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = m_RenderedShadowMaps.contains(resource)
        ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    cmd->ResourceBarrier(1, &barrier);

    const auto& light = frame.lightInfo.lights[light_index];
    const bool cube = light.lightType == LightType::Omni;
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    dsv.Texture2DArray.FirstArraySlice = cube ? layer_index * 6 : layer_index;
    dsv.Texture2DArray.ArraySize = cube ? 6 : 1;
    const auto handle = m_pShadowDsvHeap->GetCPUDescriptorHandleForHeapStart();
    rhi.GetDevice()->CreateDepthStencilView(resource, &dsv, handle);
    cmd->OMSetRenderTargets(0, nullptr, FALSE, &handle);
    cmd->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    D3D12_VIEWPORT viewport{0, 0, static_cast<float>(pShadowmap->width),
                            static_cast<float>(pShadowmap->height), 0, 1};
    D3D12_RECT scissor{0, 0, static_cast<LONG>(pShadowmap->width),
                       static_cast<LONG>(pShadowmap->height)};
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    ShadowMapConstants constants{};
    constants.light_index = light_index;
    constants.shadowmap_layer_index = static_cast<float>(layer_index);
    constants.near_plane = 1.0f;
    constants.far_plane = POINT_SHADOW_FAR_PLANE;
    if (cube) {
        const Vector3f position{light.lightPosition[0], light.lightPosition[1], light.lightPosition[2]};
        BuildD3d12PointShadowMatrices(constants.shadowMatrices, position,
                                      light.lightProjectionMatrix);
    }
    void* data = nullptr;
    D3D12_RANGE readRange{0, 0};
    auto* buffer = m_ShadowBuffers[frame.frameIndex].buffer;
    my_assert(SUCCEEDED(buffer->Map(0, &readRange, &data)));
    memcpy(static_cast<uint8_t*>(data) + light_index * kSizeShadowMapConstantBuffer,
           &constants, sizeof(constants));
    buffer->Unmap(0, nullptr);
    m_currentShadowLight = light_index;
}

void D3d12GraphicsManager::EndShadowMap(const TextureBase* pShadowmap,
                                        const int32_t layer_index, const Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* resource = reinterpret_cast<ID3D12Resource*>(pShadowmap->buff);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    rhi.GetGraphicsCommandList()->ResourceBarrier(1, &barrier);
    m_RenderedShadowMaps.insert(resource);
    m_currentShadowLight = -1;
}

void D3d12GraphicsManager::SetShadowMaps(const Frame& frame) {
    auto* cmd = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().GetGraphicsCommandList();
    const TextureBase* textures[] = {&frame.frameContext.shadowMap, &frame.frameContext.globalShadowMap,
                                     &frame.frameContext.cubeShadowMap, &frame.skybox};
    for (const auto* texture : textures) {
        auto* resource = reinterpret_cast<ID3D12Resource*>(texture->buff);
        if (resource && m_RenderedShadowMaps.insert(resource).second) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resource;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            cmd->ResourceBarrier(1, &barrier);
        }
    }
}

void D3d12GraphicsManager::CreateTexture(SceneObjectTexture& texture) {
    if (m_materialTextures.find(texture.GetName()) != m_materialTextures.end()) return;
    auto image = texture.GetTextureImage();
    if (!image) {
        std::cerr << "Could not load material texture: " << texture.GetName() << std::endl;
        return;
    }
    auto uploaded = UploadTexture(*image);
    if (image->compressed &&
        (uploaded.format == DXGI_FORMAT_BC1_UNORM ||
         uploaded.format == DXGI_FORMAT_BC2_UNORM ||
         uploaded.format == DXGI_FORMAT_BC3_UNORM ||
         uploaded.format == DXGI_FORMAT_BC4_UNORM ||
         uploaded.format == DXGI_FORMAT_BC5_UNORM ||
         uploaded.format == DXGI_FORMAT_BC7_UNORM)) {
        auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
        if (!m_pMipmapPipeline) {
            AssetLoader loader;
            auto shader = loader.SyncOpenAndReadBinary("Shaders/HLSL/generate_mips.comp.cso");
            my_assert(shader.GetData() && shader.GetDataSize());
            D3D12_SHADER_BYTECODE code{shader.GetData(), shader.GetDataSize()};
            m_pMipmapRoot = rhi.CreateRootSignature(code);
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = m_pMipmapRoot;
            desc.CS = code;
            m_pMipmapPipeline = rhi.CreateComputePipeline(desc);
        }
        auto* source = reinterpret_cast<ID3D12Resource*>(uploaded.buff);
        auto* mipmapped = rhi.GenerateTextureMipmaps(source, m_pMipmapPipeline, m_pMipmapRoot);
        SafeRelease(&source);
        uploaded.buff = reinterpret_cast<TextureHandler>(mipmapped);
        uploaded.format = mipmapped->GetDesc().Format;
        uploaded.mips = mipmapped->GetDesc().MipLevels;
    }
    m_materialTextures.emplace(texture.GetName(), uploaded);
    m_Textures.push_back(uploaded);
}

void D3d12GraphicsManager::ReleaseTexture(TextureBase& texture) {
    ID3D12Resource* pTmp = reinterpret_cast<ID3D12Resource*>(texture.buff);
    auto debugView = m_debugViews.find(pTmp);
    if (debugView != m_debugViews.end()) {
        dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().WaitForIdle();
        SafeRelease(&debugView->second.source);
        m_debugViews.erase(debugView);
    }
    auto view = m_renderTextures.find(pTmp);
    if (view != m_renderTextures.end()) {
        SafeRelease(&view->second.viewHeap);
        m_renderTextures.erase(view);
    }
    auto slot = m_imguiSlots.find(texture.buff);
    if (slot != m_imguiSlots.end()) {
        m_freeImGuiSlots.push_back(slot->second);
        m_imguiSlots.erase(slot);
    }
    SafeRelease(&pTmp);
    texture.buff = 0;
    texture.handler = 0;
}

void D3d12GraphicsManager::GenerateTextureForWrite(Texture2D& texture) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto* device = rhi.GetDevice();
    my_assert(texture.pixel_format == PIXEL_FORMAT::RG32);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = texture.width;
    desc.Height = texture.height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32G32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* resource = nullptr;
    my_assert(SUCCEEDED(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&resource))));
    texture.format = desc.Format;
    texture.buff = reinterpret_cast<TextureHandler>(resource);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = desc.Format;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    CreateImGuiSrv(texture, srv);
    m_Textures.push_back(texture);
}

void D3d12GraphicsManager::BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) {
    my_assert(slot_index == 0 && texture.buff);
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    m_pComputeTexture = reinterpret_cast<ID3D12Resource*>(texture.buff);
    rhi.BindComputeTexture(m_pComputeTexture, m_pComputeHeap);
}

void D3d12GraphicsManager::BindTexture(Texture2D& texture, const uint32_t slot_index) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    my_assert(slot_index < 2);
    rhi.CreateDescriptorSet(m_pCbvSrvUavHeap, slot_index, (ID3D12Resource**)&texture.buff, 1);
    texture.handler = (TextureHandler)rhi.GetGpuDescriptorHandle(m_pCbvSrvUavHeap, slot_index).ptr;
}

void D3d12GraphicsManager::BindDebugTexture(Texture2D& texture, const uint32_t slot_index) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.CreateDescriptorSet(m_pCbvSrvUavHeapImGui, slot_index, (ID3D12Resource**)&texture.buff, 1);
    texture.handler = (TextureHandler)rhi.GetGpuDescriptorHandle(m_pCbvSrvUavHeapImGui, slot_index).ptr;
}

void D3d12GraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                    const uint32_t depth) {
    dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().Dispatch(width, height, depth);
}

void D3d12GraphicsManager::BeginCompute() {
    dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().BeginCompute();
}

void D3d12GraphicsManager::EndCompute() {
    my_assert(m_pComputeTexture);
    dynamic_cast<D3d12Application*>(m_pApp)->GetRHI().EndCompute(m_pComputeTexture);
    m_pComputeTexture = nullptr;
}
