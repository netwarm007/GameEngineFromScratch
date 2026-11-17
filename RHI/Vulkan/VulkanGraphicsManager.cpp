#include "VulkanGraphicsManager.hpp"
#include "VulkanShaderBindings.hpp"

#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

using namespace My;

VulkanGraphicsManager::~VulkanGraphicsManager() {
}

int VulkanGraphicsManager::Initialize() {
    if (!m_rhi.GetDevice()) {
        throw std::logic_error("Vulkan graphics initialization requires an initialized RHI");
    }
    return GraphicsManager::Initialize();
}

void VulkanGraphicsManager::Finalize() {
    m_InitPasses.clear();
    m_DispatchPasses.clear();
    m_DrawPasses.clear();
    GraphicsManager::Finalize();
    m_ownedTextures.clear();
    m_nSceneRevision = 0;
    m_nFrameIndex = 0;
}

size_t VulkanGraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
    const void* pData = v_property_array.GetData();
    auto size = v_property_array.GetDataSize();
    auto stride = size / v_property_array.GetVertexCount();

    auto& rhi = m_rhi;

    return 0;
}

size_t VulkanGraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
    const void* pData = index_array.GetData();
    auto size = index_array.GetDataSize();
    int32_t index_size =
        static_cast<int32_t>(size / index_array.GetIndexCount());

    auto& rhi = m_rhi;

    return 0;
}

void VulkanGraphicsManager::initializeGeometries(const Scene& scene) {
    std::cout << "Creating Draw Batch Contexts ...";
    uint32_t batch_index = 0;
    for (const auto& _it : scene.GeometryNodes) {
        const auto& pGeometryNode = _it.second.lock();

        if (pGeometryNode) {
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
    auto& rhi = m_rhi;

    assert(scene.SkyBox);
}

void VulkanGraphicsManager::EndScene() {
    m_rhi.WaitGPU();
    m_computeActive = false;
    m_computePipeline.reset();
    m_computeOutput = 0;
    std::set<TextureHandler> sharedTextures;
    for (auto& frame : m_Frames) {
        for (auto* texture : std::initializer_list<TextureBase*>{
                 &frame.brdfLUT, &frame.skybox, &frame.terrainHeightMap,
                 &frame.frameContext.shadowMap, &frame.frameContext.globalShadowMap,
                 &frame.frameContext.cubeShadowMap}) {
            if (texture->handler) sharedTextures.insert(texture->handler);
        }
    }
    for (const auto handle : sharedTextures) {
        TextureBase texture;
        texture.handler = handle;
        ReleaseTexture(texture);
    }
    GraphicsManager::EndScene();
    m_Textures.clear();
    material_map.clear();
    for (auto& frame : m_Frames) {
        const auto index = frame.frameIndex;
        frame = Frame{};
        frame.frameIndex = index;
    }
}

void VulkanGraphicsManager::BeginFrame(Frame& frame) {
    GraphicsManager::BeginFrame(frame);

    auto& rhi = m_rhi;

    assert(frame.frameIndex == m_nFrameIndex);
}

void VulkanGraphicsManager::EndFrame(Frame& frame) {
    auto& rhi = m_rhi;

    GraphicsManager::EndFrame(frame);
}

void VulkanGraphicsManager::BeginPass(Frame& frame) {
    auto& rhi = m_rhi;
}

void VulkanGraphicsManager::EndPass(Frame& frame) {
    auto& rhi = m_rhi;
}

void VulkanGraphicsManager::DrawBatch(const Frame& frame) {
    for (const auto& pDbc : frame.batchContexts) {
        if (!pDbc->node->Visible()) continue;
        const VulkanDrawBatchContext& dbc =
            dynamic_cast<const VulkanDrawBatchContext&>(*pDbc);
    }

    auto& rhi = m_rhi;
}

void VulkanGraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    auto state = std::dynamic_pointer_cast<VulkanPipelineState>(pipelineState);
    if (!m_computeActive || !state || state->pipelineType != PIPELINE_TYPE::COMPUTE ||
        !state->pipelineState || !state->pipelineLayout || !state->descriptorSetLayout) {
        throw std::logic_error("Vulkan compute binding requires a live compute pipeline and pass");
    }
    if (state->computeShaderName != "integrateBRDF.comp") {
        throw std::invalid_argument("Vulkan graphics compute binding currently supports the shared BRDF kernel only");
    }
    m_computePipeline = std::move(state);
}

void VulkanGraphicsManager::SetPerFrameConstants(const Frame& frame) {}

void VulkanGraphicsManager::SetLightInfo(const Frame& frame) {}

void VulkanGraphicsManager::Present() {
    auto& rhi = m_rhi;
}

void VulkanGraphicsManager::DrawSkyBox(const Frame& frame) {}

void VulkanGraphicsManager::GenerateCubemapArray(TextureCubeArray& texture_array) {
    if (texture_array.size > (std::numeric_limits<uint32_t>::max)() / 6) {
        throw std::invalid_argument("Vulkan cube array layer count overflows");
    }
    if (texture_array.pixel_format != PIXEL_FORMAT::D32) {
        throw std::invalid_argument("Vulkan cube-array generation currently requires D32");
    }
    GenerateTextureResource(texture_array, texture_array.size * 6,
        vk::ImageViewType::eCubeArray,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::Format::eD32Sfloat);
}

void VulkanGraphicsManager::GenerateTextureArray(Texture2DArray& texture_array) {
    if (texture_array.pixel_format != PIXEL_FORMAT::D32) {
        throw std::invalid_argument("Vulkan shadow-array generation requires D32");
    }
    GenerateTextureResource(texture_array, texture_array.size,
        vk::ImageViewType::e2DArray,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::Format::eD32Sfloat);
}

void VulkanGraphicsManager::BeginShadowMap(
    const int32_t light_index, const TextureBase* pShadowmap, const int32_t layer_index, const Frame& frame) {}

void VulkanGraphicsManager::EndShadowMap(const TextureBase* pShadowmap,
                                        const int32_t layer_index, const Frame& frame) {}

void VulkanGraphicsManager::SetShadowMaps(const Frame& frame) {}

void VulkanGraphicsManager::CreateTexture(SceneObjectTexture& texture) {}

void VulkanGraphicsManager::ReleaseTexture(TextureBase& texture) {
    if (!texture.handler && !texture.buff) return;
    GetTextureResource(texture);
    m_rhi.WaitGPU();
    if (m_computeOutput == texture.handler) m_computeOutput = 0;
    m_ownedTextures.erase(texture.handler);
    texture.handler = 0;
    texture.buff = 0;
}

void VulkanGraphicsManager::GenerateTextureResource(
    TextureBase& texture, uint32_t layers, vk::ImageViewType viewType,
    vk::ImageUsageFlags usage, vk::Format format) {
    if (texture.handler || texture.buff || texture.samples != 1 ||
        m_nextTexture == (std::numeric_limits<TextureHandler>::max)()) {
        throw std::invalid_argument("Vulkan texture generation requires an empty, single-sample descriptor");
    }
    auto owned = std::make_unique<OwnedTexture>(m_rhi);
    owned->texture = m_rhi.CreateTexture(texture.width, texture.height, format,
        texture.mips, layers, viewType, usage);
    const auto id = m_nextTexture + 1;
    m_ownedTextures.emplace(id, std::move(owned));
    m_nextTexture = id;
    texture.handler = id;
}

VulkanRHI::Texture& VulkanGraphicsManager::GetTextureResource(const TextureBase& texture) {
    const auto found = m_ownedTextures.find(texture.handler);
    if (texture.buff || !texture.handler || found == m_ownedTextures.end()) {
        throw std::invalid_argument("Vulkan texture descriptor does not name a live owned texture");
    }
    return found->second->texture;
}

void VulkanGraphicsManager::GenerateTexture(Texture2D& texture) {
    vk::Format format;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled;
    switch (texture.pixel_format) {
        case PIXEL_FORMAT::RGBA8:
            format = vk::Format::eR8G8B8A8Unorm;
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            break;
        case PIXEL_FORMAT::D32:
            format = vk::Format::eD32Sfloat;
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
            break;
        default: throw std::invalid_argument("Unsupported Vulkan render-target pixel format");
    }
    GenerateTextureResource(texture, 1, vk::ImageViewType::e2D, usage, format);
}

void VulkanGraphicsManager::GenerateTextureForWrite(Texture2D& texture) {
    if ((texture.pixel_format != PIXEL_FORMAT::RG32 &&
         texture.pixel_format != PIXEL_FORMAT::RGBA32) || texture.mips != 1) {
        throw std::invalid_argument("Vulkan shared compute output requires a single-mip floating-point texture");
    }
    // Shared HLSL declares RWTexture2D<float4>, reflected as rgba32f in SPIR-V.
    GenerateTextureResource(texture, 1, vk::ImageViewType::e2D,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
        vk::Format::eR32G32B32A32Sfloat);
}

void VulkanGraphicsManager::BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) {
    if (!m_computeActive) throw std::logic_error("Vulkan storage binding requires an active compute pass");
    const auto& resource = GetTextureResource(texture);
    if (slot_index != 0 || !(resource.usage & vk::ImageUsageFlagBits::eStorage)) {
        throw std::invalid_argument("Vulkan shared compute output requires storage slot zero");
    }
    m_computeOutput = texture.handler;
}

void VulkanGraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                    const uint32_t depth) {
    if (!m_computeActive || !m_computePipeline || !m_computePipeline->pipelineState ||
        !m_computeOutput) {
        throw std::logic_error("Vulkan dispatch requires an active pass, live pipeline, and storage output");
    }
    auto& output = m_ownedTextures.at(m_computeOutput)->texture;
    if (width != 512 || height != 512 || depth != 1 ||
        output.width != width || output.height != height) {
        throw std::invalid_argument("Shared BRDF dispatch requires exactly a 512-by-512 output");
    }
    const auto limits = m_rhi.GetPhysicalDevice().getProperties().limits;
    if (width > limits.maxComputeWorkGroupCount[0] ||
        height > limits.maxComputeWorkGroupCount[1]) {
        throw std::runtime_error("Vulkan device cannot dispatch the shared BRDF workgroup grid");
    }
    const auto device = m_rhi.GetDevice();
    const std::array sizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 4),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 12),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1)};
    const auto pool = device.createDescriptorPoolUnique({{}, 1, sizes});
    const auto set = device.allocateDescriptorSets(
        {*pool, *m_computePipeline->descriptorSetLayout}).front();
    const vk::DescriptorImageInfo image({}, output.descriptor, vk::ImageLayout::eGeneral);
    device.updateDescriptorSets(vk::WriteDescriptorSet(set,
        VulkanShaderBindings::kStorageImageBase, 0, 1,
        vk::DescriptorType::eStorageImage, &image), {});
    m_rhi.ExecuteCommands([&](vk::CommandBuffer command) {
        const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        m_rhi.TransitionTexture(command, output, range, vk::ImageLayout::eGeneral);
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline->pipelineState);
        const std::array<uint32_t, 4> offsets{};
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
            *m_computePipeline->pipelineLayout, 0, set, offsets);
        command.dispatch(width, height, depth);
        m_rhi.TransitionTexture(command, output, range, vk::ImageLayout::eShaderReadOnlyOptimal);
    });
}

std::vector<uint8_t> VulkanGraphicsManager::ReadTexture(const TextureBase& texture) {
    return m_rhi.ReadTexture(GetTextureResource(texture));
}

void VulkanGraphicsManager::BeginCompute() {
    if (!IsInitialized() || m_computeActive) {
        throw std::logic_error("Vulkan compute pass requires an initialized manager and no active pass");
    }
    m_computePipeline.reset();
    m_computeOutput = 0;
    m_computeActive = true;
}

void VulkanGraphicsManager::EndCompute() {
    if (!m_computeActive) throw std::logic_error("No Vulkan compute pass is active");
    m_computePipeline.reset();
    m_computeOutput = 0;
    m_computeActive = false;
}
