#include "D3d12GraphicsManager.hpp"

#include <objbase.h>

#include <iostream>

#include "AssetLoader.hpp"
#include "D3d12Application.hpp"
#include "D3d12Utility.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"

#include "D3d12RHI.hpp"

#include "imgui_impl_dx12.h"
#include "imgui_impl_win32.h"

using namespace My;
using namespace std;

D3d12GraphicsManager::~D3d12GraphicsManager() {
    SafeRelease(&m_pCbvSrvUavHeapImGui);
}

int D3d12GraphicsManager::Initialize() {
    int result = GraphicsManager::Initialize();

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    auto pDev = rhi.GetDevice();

    D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc{};
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvSrvUavHeapDesc.NumDescriptors = 1;

    assert(SUCCEEDED(pDev->CreateDescriptorHeap(
        &cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_pCbvSrvUavHeapImGui))));
    m_pCbvSrvUavHeapImGui->SetName(L"ImGui CbvSrvUav Heap");

    auto cpuDescriptorHandle =
        m_pCbvSrvUavHeapImGui->GetCPUDescriptorHandleForHeapStart();
    auto gpuDescriptorHandle =
        m_pCbvSrvUavHeapImGui->GetGPUDescriptorHandleForHeapStart();

    ImGui_ImplDX12_Init(pDev, GfxConfiguration::kMaxInFlightFrameCount,
                        ::DXGI_FORMAT_R8G8B8A8_UNORM, m_pCbvSrvUavHeapImGui,
                        cpuDescriptorHandle, gpuDescriptorHandle);

    // 创建命令清单池
    rhi.CreateCommandPools();

    // 创建命令列表
    rhi.CreateCommandLists();

    return result;
}

void D3d12GraphicsManager::Finalize() {
    ImGui_ImplDX12_Shutdown();

    GraphicsManager::Finalize();
}

size_t D3d12GraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
    const void* pData = v_property_array.GetData();
    auto size = v_property_array.GetDataSize();
    auto stride = size / v_property_array.GetVertexCount();

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    return rhi.CreateVertexBuffer(pData, size, (int32_t)stride);
}

size_t D3d12GraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
    const void* pData = index_array.GetData();
    auto size = index_array.GetDataSize();
    int32_t index_size =
        static_cast<int32_t>(size / index_array.GetIndexCount());

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    return rhi.CreateIndexBuffer(pData, size, index_size);
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
                assert(0);
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
                    assert(0);
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
                assert(0);
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
            psod.SampleDesc.Count = 4;  // 4X MSAA
            psod.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
        }
        psod.DSVFormat = ::DXGI_FORMAT_D32_FLOAT;

        pipelineState.pipelineState = rhi.CreateGraphicsPipeline(psod);
    } else {
        assert(pipelineState.pipelineType == PIPELINE_TYPE::COMPUTE);

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
            assert(pGeometry);
            const auto& pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            // Set the number of vertex properties.
            const auto vertexPropertiesCount =
                pMesh->GetVertexPropertiesCount();

            // Set the number of vertices in the vertex array.
            const auto vertexCount = pMesh->GetVertexCount();

            auto dbc = make_shared<D3dDrawBatchContext>();

            for (uint32_t i = 0; i < vertexPropertiesCount; i++) {
                const SceneObjectVertexArray& v_property_array =
                    pMesh->GetVertexPropertyArray(i);

                auto offset = CreateVertexBuffer(v_property_array);
                if (i == 0) {
                    dbc->property_offset = offset;
                }
            }

            const SceneObjectIndexArray& index_array = pMesh->GetIndexArray(0);
            dbc->index_offset = CreateIndexBuffer(index_array);

            const auto material_index = index_array.GetMaterialIndex();
            const auto material_key =
                pGeometryNode->GetMaterialRef(material_index);
            const auto& material = scene.GetMaterial(material_key);

            dbc->batchIndex = batch_index++;
            dbc->index_count = (UINT)index_array.GetIndexCount();
            dbc->property_count = vertexPropertiesCount;

#if 0
            // load material textures
            dbc->cbv_srv_uav_offset =
                (size_t)dbc->batchIndex * 32 * m_nCbvSrvUavDescriptorSize;
            D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle =
                m_pCbvSrvUavHeaps[0]->GetCPUDescriptorHandleForHeapStart();
            srvCpuHandle.ptr += dbc->cbv_srv_uav_offset;

            // Jump over per batch CBVs
            srvCpuHandle.ptr += 2 * m_nCbvSrvUavDescriptorSize;

            // SRV
            if (material) {
                if (auto& texture = material->GetBaseColor().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetNormal().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetMetallic().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetRoughness().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetAO().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }
            }

#endif

            // UAV
            // ; temporary nothing here

            dbc->node = pGeometryNode;

            for (auto& frame : m_Frames) {
                frame.batchContexts.push_back(dbc);
            }
        }
    }
    cout << "Done!" << endl;
}

void D3d12GraphicsManager::initializeSkyBox(const Scene& scene) {
    HRESULT hr = S_OK;

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    assert(scene.SkyBox);

    m_dbcSkyBox.property_offset =
        rhi.CreateVertexBuffer(SceneObjectSkyBox::skyboxVertices,
                               sizeof(SceneObjectSkyBox::skyboxVertices),
                               sizeof(SceneObjectSkyBox::skyboxVertices[0]));
    m_dbcSkyBox.property_count = 1;
    m_dbcSkyBox.index_offset = rhi.CreateIndexBuffer(
        SceneObjectSkyBox::skyboxIndices,
        sizeof(SceneObjectSkyBox::skyboxIndices),
        static_cast<int32_t>(sizeof(SceneObjectSkyBox::skyboxIndices[0])));
    m_dbcSkyBox.index_count = sizeof(SceneObjectSkyBox::skyboxIndices) /
                              sizeof(SceneObjectSkyBox::skyboxIndices[0]);

    // Describe and create a Cubemap.
    auto& texture = scene.SkyBox->GetTexture(0);
    const auto& pImage = texture.GetTextureImage();
    rhi.CreateTextureImage(*pImage);
}

void D3d12GraphicsManager::EndScene() {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();

    rhi.ResetAllBuffers();

    GraphicsManager::EndScene();
}

void D3d12GraphicsManager::BeginFrame(Frame& frame) {
    GraphicsManager::BeginFrame(frame);
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.BeginFrame();

    assert(frame.frameIndex == m_nFrameIndex);
}

void D3d12GraphicsManager::EndFrame(Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.EndFrame();
    rhi.DrawGUI(m_pCbvSrvUavHeapImGui);
    GraphicsManager::EndFrame(frame);
}

void D3d12GraphicsManager::BeginPass(Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.BeginPass(frame.clearColor);
}

void D3d12GraphicsManager::EndPass(Frame& frame) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.EndPass();
}

void D3d12GraphicsManager::DrawBatch(const Frame& frame) {
    for (const auto& pDbc : frame.batchContexts) {
        const D3dDrawBatchContext& dbc =
            dynamic_cast<const D3dDrawBatchContext&>(*pDbc);
    }

    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.Draw();
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

        rhi.SetPipelineState(pState->pipelineState);
    }
}

void D3d12GraphicsManager::SetPerFrameConstants(const Frame& frame) {}

void D3d12GraphicsManager::SetLightInfo(const Frame& frame) {}

void D3d12GraphicsManager::Present() {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
    rhi.Present();
}

void D3d12GraphicsManager::DrawSkyBox(const Frame& frame) {}

void D3d12GraphicsManager::GenerateCubemapArray(TextureCubeArray& texture_array) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
}

void D3d12GraphicsManager::GenerateTextureArray(Texture2DArray& texture_array) {
    auto& rhi = dynamic_cast<D3d12Application*>(m_pApp)->GetRHI();
}

void D3d12GraphicsManager::BeginShadowMap(
    const int32_t light_index, const TextureBase* pShadowmap, const int32_t layer_index, const Frame& frame) {}

void D3d12GraphicsManager::EndShadowMap(const TextureBase* pShadowmap,
                                        const int32_t layer_index, const Frame& frame) {}

void D3d12GraphicsManager::SetShadowMaps(const Frame& frame) {}

void D3d12GraphicsManager::CreateTexture(SceneObjectTexture& texture) {}

void D3d12GraphicsManager::ReleaseTexture(TextureBase& texture) {
    ID3D12Resource* pTmp = reinterpret_cast<ID3D12Resource*>(texture.handler);
    SafeRelease(&pTmp);
}

void D3d12GraphicsManager::GenerateTextureForWrite(Texture2D& texture) {}

void D3d12GraphicsManager::BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) {}

void D3d12GraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                    const uint32_t depth) {}

void D3d12GraphicsManager::BeginCompute() {}

void D3d12GraphicsManager::EndCompute() {}
