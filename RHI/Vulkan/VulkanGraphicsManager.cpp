#include "VulkanGraphicsManager.hpp"
#include "VulkanApplication.hpp"

#include <memory>

using namespace My;

VulkanGraphicsManager::~VulkanGraphicsManager() {
}

int VulkanGraphicsManager::Initialize() {
    int result = GraphicsManager::Initialize();

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    return result;
}

void VulkanGraphicsManager::Finalize() {
    GraphicsManager::Finalize();
}

size_t VulkanGraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
    const void* pData = v_property_array.GetData();
    auto size = v_property_array.GetDataSize();
    auto stride = size / v_property_array.GetVertexCount();

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    return 0;
}

size_t VulkanGraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
    const void* pData = index_array.GetData();
    auto size = index_array.GetDataSize();
    int32_t index_size =
        static_cast<int32_t>(size / index_array.GetIndexCount());

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    return 0;
}

void VulkanGraphicsManager::initializeGeometries(const Scene& scene) {
    std::cout << "Creating Draw Batch Contexts ...";
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

            auto dbc = std::make_shared<VulkanDrawBatchContext>();

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
            dbc->index_count = (uint32_t)index_array.GetIndexCount();
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
    std::cout << "Done!" << std::endl;
}

void VulkanGraphicsManager::initializeSkyBox(const Scene& scene) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    assert(scene.SkyBox);
}

void VulkanGraphicsManager::EndScene() {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    GraphicsManager::EndScene();
}

void VulkanGraphicsManager::BeginFrame(Frame& frame) {
    GraphicsManager::BeginFrame(frame);

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    assert(frame.frameIndex == m_nFrameIndex);
}

void VulkanGraphicsManager::EndFrame(Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    GraphicsManager::EndFrame(frame);
}

void VulkanGraphicsManager::BeginPass(Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
}

void VulkanGraphicsManager::EndPass(Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
}

void VulkanGraphicsManager::DrawBatch(const Frame& frame) {
    for (const auto& pDbc : frame.batchContexts) {
        const VulkanDrawBatchContext& dbc =
            dynamic_cast<const VulkanDrawBatchContext&>(*pDbc);
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
}

void VulkanGraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    if (pipelineState) {
        std::shared_ptr<VulkanPipelineState> pState =
            dynamic_pointer_cast<VulkanPipelineState>(pipelineState);
    }
}

void VulkanGraphicsManager::SetPerFrameConstants(const Frame& frame) {}

void VulkanGraphicsManager::SetLightInfo(const Frame& frame) {}

void VulkanGraphicsManager::Present() {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
}

void VulkanGraphicsManager::DrawSkyBox(const Frame& frame) {}

void VulkanGraphicsManager::GenerateCubemapArray(TextureCubeArray& texture_array) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
}

void VulkanGraphicsManager::GenerateTextureArray(Texture2DArray& texture_array) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
}

void VulkanGraphicsManager::BeginShadowMap(
    const int32_t light_index, const TextureBase* pShadowmap, const int32_t layer_index, const Frame& frame) {}

void VulkanGraphicsManager::EndShadowMap(const TextureBase* pShadowmap,
                                        const int32_t layer_index, const Frame& frame) {}

void VulkanGraphicsManager::SetShadowMaps(const Frame& frame) {}

void VulkanGraphicsManager::CreateTexture(SceneObjectTexture& texture) {}

void VulkanGraphicsManager::ReleaseTexture(TextureBase& texture) {
}

void VulkanGraphicsManager::GenerateTextureForWrite(Texture2D& texture) {}

void VulkanGraphicsManager::BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) {}

void VulkanGraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                    const uint32_t depth) {}

void VulkanGraphicsManager::BeginCompute() {}

void VulkanGraphicsManager::EndCompute() {}
