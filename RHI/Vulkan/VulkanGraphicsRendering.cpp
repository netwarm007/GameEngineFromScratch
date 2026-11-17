#include "VulkanGraphicsManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "VulkanShaderBindings.hpp"

using namespace My;

namespace {
uint8_t Channel(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Non-finite material parameter");
    return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}
vk::ImageSubresourceRange Whole(const VulkanRHI::Texture& texture) {
    return {texture.aspect, 0, texture.mipLevels, 0, texture.layers};
}
}  // namespace

vk::Buffer VulkanGraphicsManager::UploadVertex(const void* data, size_t size) {
    VulkanRHI::VertexBuffer result;
    m_rhi.createVertexBuffer(data, size, result);
    BufferResource owned{vk::UniqueDeviceMemory(result.heap, m_rhi.GetDevice()),
                         vk::UniqueBuffer(result.buffer, m_rhi.GetDevice())};
    m_buffers.push_back(std::move(owned));
    return result.buffer;
}

vk::Buffer VulkanGraphicsManager::UploadIndex(const void* data, size_t size) {
    VulkanRHI::IndexBuffer result;
    m_rhi.createIndexBuffer(data, size, result);
    BufferResource owned{vk::UniqueDeviceMemory(result.heap, m_rhi.GetDevice()),
                         vk::UniqueBuffer(result.buffer, m_rhi.GetDevice())};
    m_buffers.push_back(std::move(owned));
    return result.buffer;
}

Texture2D VulkanGraphicsManager::CreateTexture(Image& image) {
    if (m_nextTexture == (std::numeric_limits<TextureHandler>::max)()) {
        throw std::overflow_error("Vulkan texture identity exhausted");
    }
    auto resource = std::make_unique<OwnedTexture>(m_rhi);
    resource->texture = m_rhi.createTextureImage(image);
    Texture2D texture;
    texture.width = image.Width;
    texture.height = image.Height;
    texture.mips = resource->texture.mipLevels;
    texture.pixel_format = image.pixel_format;
    texture.format = static_cast<TextureFormat>(resource->texture.format);
    const auto id = m_nextTexture + 1;
    m_ownedTextures.emplace(id, std::move(resource));
    m_nextTexture = texture.handler = id;
    return texture;
}

Texture2D VulkanGraphicsManager::SolidTexture(const std::array<uint8_t, 4>& pixel) {
    Image image;
    image.Width = image.Height = 1;
    image.bitcount = 32;
    image.bitdepth = 8;
    image.pitch = image.data_size = 4;
    image.pixel_format = PIXEL_FORMAT::RGBA8;
    image.data = new uint8_t[4];
    std::memcpy(image.data, pixel.data(), 4);
    return CreateTexture(image);
}

Texture2D VulkanGraphicsManager::MaterialTexture(
    const std::shared_ptr<SceneObjectTexture>& source, const std::array<uint8_t, 4>& value) {
    if (source) {
        CreateTexture(*source);
        return m_materialTextures.at(source.get());
    }
    auto texture = SolidTexture(value);
    m_Textures.push_back(texture);
    return texture;
}

void VulkanGraphicsManager::CreateTexture(SceneObjectTexture& source) {
    if (m_materialTextures.contains(&source)) return;
    const auto image = source.GetTextureImage();
    if (!image) throw std::runtime_error("Vulkan material image was not loaded: " + source.GetName());
    auto texture = CreateTexture(*image);
    m_Textures.push_back(texture);
    m_materialTextures.emplace(&source, texture);
}

void VulkanGraphicsManager::InitializeDepth(TextureBase& descriptor) {
    auto& texture = GetTextureResource(descriptor);
    m_rhi.ExecuteCommands([&](vk::CommandBuffer command) {
        m_rhi.TransitionTexture(command, texture, Whole(texture), vk::ImageLayout::eTransferDstOptimal);
        command.clearDepthStencilImage(texture.image, vk::ImageLayout::eTransferDstOptimal,
                                        vk::ClearDepthStencilValue(1.0f, 0), Whole(texture));
        m_rhi.TransitionTexture(command, texture, Whole(texture),
                                vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    });
}

void VulkanGraphicsManager::InitializeFallbacks() {
    m_fallback2D[0] = SolidTexture({255, 255, 255, 255});
    m_fallback2D[1] = SolidTexture({128, 128, 255, 255});
    m_fallback2D[2] = SolidTexture({0, 255, 255, 255});
    m_fallback2D[3] = SolidTexture({0, 0, 0, 255});
    m_fallbackShadow.width = m_fallbackShadow.height = m_fallbackShadow.size = 1;
    m_fallbackShadow.pixel_format = PIXEL_FORMAT::D32;
    GenerateTextureArray(m_fallbackShadow);
    m_fallbackCubeShadow.width = m_fallbackCubeShadow.height = m_fallbackCubeShadow.size = 1;
    m_fallbackCubeShadow.pixel_format = PIXEL_FORMAT::D32;
    GenerateCubemapArray(m_fallbackCubeShadow);
    m_fallbackSky.width = m_fallbackSky.height = 1;
    m_fallbackSky.size = 2;
    m_fallbackSky.pixel_format = PIXEL_FORMAT::RGBA8;
    GenerateTextureResource(m_fallbackSky, 12, vk::ImageViewType::eCubeArray,
        vk::ImageUsageFlagBits::eSampled, vk::Format::eR8G8B8A8Unorm);
    auto& sky = GetTextureResource(m_fallbackSky);
    const std::array<uint8_t, 4> black{0, 0, 0, 255};
    for (uint32_t layer = 0; layer < sky.layers; ++layer) {
        m_rhi.UpdateTexture(sky, black.data(), black.size(), 0, layer);
    }
    vk::SamplerCreateInfo sampler;
    const auto linear = m_rhi.GetPhysicalDevice().getFormatProperties(vk::Format::eD32Sfloat)
                            .optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
    if (!linear) {
        std::cerr << "[Vulkan] D32 linear filtering unavailable; shadows use nearest filtering\n";
    }
    sampler.minFilter = sampler.magFilter = vk::Filter::eLinear;
    sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eRepeat;
    sampler.maxLod = VK_LOD_CLAMP_NONE;
    m_sampler = m_rhi.GetDevice().createSamplerUnique(sampler);
    sampler.minFilter = sampler.magFilter = linear ? vk::Filter::eLinear : vk::Filter::eNearest;
    sampler.mipmapMode = vk::SamplerMipmapMode::eNearest;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    m_shadowSampler = m_rhi.GetDevice().createSamplerUnique(sampler);
}

void VulkanGraphicsManager::initializeGeometries(const Scene& scene) {
    uint32_t batchIndex = 0;
    for (const auto& [name, weakNode] : scene.GeometryNodes) {
        const auto node = weakNode.lock();
        if (!node) continue;
        const auto geometry = scene.GetGeometry(node->GetSceneObjectRef());
        if (!geometry) throw std::runtime_error("Vulkan geometry reference is unresolved: " + name);
        const auto mesh = geometry->GetMesh().lock();
        if (!mesh) continue;
        if (mesh->GetPrimitiveType() != PrimitiveType::kPrimitiveTypeTriList ||
            mesh->GetVertexCount() == 0) {
            throw std::invalid_argument("Vulkan scene meshes require nonempty triangle lists");
        }
        std::array<vk::Buffer, 4> vertices{};
        const std::array names{"position", "normal", "texcoord", "tangent"};
        const std::array types{VertexDataType::kVertexDataTypeFloat3, VertexDataType::kVertexDataTypeFloat3,
                              VertexDataType::kVertexDataTypeFloat2, VertexDataType::kVertexDataTypeFloat3};
        for (uint32_t index = 0; index < mesh->GetVertexPropertiesCount(); ++index) {
            const auto& array = mesh->GetVertexPropertyArray(index);
            const auto found = std::find(names.begin(), names.end(), array.GetAttributeName());
            if (found == names.end()) throw std::invalid_argument("Unsupported Vulkan scene vertex attribute");
            const auto slot = static_cast<size_t>(found - names.begin());
            if (vertices[slot] || array.GetDataType() != types[slot] ||
                array.GetVertexCount() != mesh->GetVertexCount()) {
                throw std::invalid_argument("Invalid Vulkan shared vertex stream layout");
            }
            vertices[slot] = UploadVertex(array.GetData(), array.GetDataSize());
        }
        if (!vertices[0]) throw std::invalid_argument("Vulkan mesh has no position stream");
        for (size_t slot = 1; slot < vertices.size(); ++slot) {
            if (!vertices[slot]) {
                const auto components = slot == 2 ? 2u : 3u;
                std::vector<float> values(mesh->GetVertexCount() * components, 0.0f);
                for (size_t vertex = 0; vertex < mesh->GetVertexCount(); ++vertex) {
                    if (slot == 1) values[vertex * components + 2] = 1.0f;
                    if (slot == 3) values[vertex * components] = 1.0f;
                }
                vertices[slot] = UploadVertex(values.data(), values.size() * sizeof(float));
            }
        }
        for (size_t group = 0; group < mesh->GetIndexGroupCount(); ++group) {
            const auto& array = mesh->GetIndexArray(group);
            if (!array.GetData() || !array.GetIndexCount() || array.GetIndexCount() % 3 ||
                array.GetIndexCount() > (std::numeric_limits<uint32_t>::max)() ||
                batchIndex >= GfxConfiguration::kMaxSceneObjectCount) {
                throw std::invalid_argument("Invalid Vulkan triangle index group or batch capacity");
            }
            std::vector<uint32_t> indices(array.GetIndexCount());
            for (size_t index = 0; index < indices.size(); ++index) {
                uint64_t value = 0;
                switch (array.GetIndexType()) {
                    case IndexDataType::kIndexDataTypeInt8: value = static_cast<const uint8_t*>(array.GetData())[index]; break;
                    case IndexDataType::kIndexDataTypeInt16: value = static_cast<const uint16_t*>(array.GetData())[index]; break;
                    case IndexDataType::kIndexDataTypeInt32: value = static_cast<const uint32_t*>(array.GetData())[index]; break;
                    case IndexDataType::kIndexDataTypeInt64: value = static_cast<const uint64_t*>(array.GetData())[index]; break;
                    default: throw std::invalid_argument("Unsupported Vulkan index type");
                }
                if (value >= mesh->GetVertexCount()) throw std::invalid_argument("Vulkan index exceeds vertex count");
                indices[index] = static_cast<uint32_t>(value);
            }
            auto batch = std::make_shared<VulkanDrawBatchContext>();
            batch->vertices = vertices;
            batch->indices = UploadIndex(indices.data(), indices.size() * sizeof(uint32_t));
            batch->indexCount = static_cast<uint32_t>(indices.size());
            batch->batchIndex = batchIndex++;
            batch->node = node;
            batch->modelMatrix = node->GetRenderTransform();
            const auto material = scene.GetMaterial(node->GetMaterialRef(array.GetMaterialIndex()));
            if (!material) throw std::runtime_error("Vulkan material reference is unresolved");
            const auto& color = material->GetBaseColor();
            batch->material.diffuseMap = MaterialTexture(color.ValueMap,
                {Channel(color.Value[0]), Channel(color.Value[1]), Channel(color.Value[2]), Channel(color.Value[3])});
            batch->material.normalMap = MaterialTexture(material->GetNormal().ValueMap, {128, 128, 255, 255});
            batch->material.metallicMap = MaterialTexture(material->GetMetallic().ValueMap,
                {Channel(material->GetMetallic().Value), Channel(material->GetRoughness().Value),
                 Channel(material->GetAO().Value), 255});
            batch->material.roughnessMap = MaterialTexture(material->GetRoughness().ValueMap,
                {Channel(material->GetRoughness().Value), 0, 0, 255});
            batch->material.aoMap = MaterialTexture(material->GetAO().ValueMap,
                {Channel(material->GetAO().Value), 0, 0, 255});
            batch->material.heightMap = MaterialTexture(material->GetHeight().ValueMap,
                {Channel(material->GetHeight().Value), 0, 0, 255});
            m_Frames.front().batchContexts.push_back(std::move(batch));
        }
    }
}

vk::CommandBuffer VulkanGraphicsManager::Command() const {
    if (!m_frameRecording) throw std::logic_error("No Vulkan graphics frame is recording");
    return m_commands ? m_commands->Get() : m_rhi.GetFrameCommandBuffer();
}

uint32_t VulkanGraphicsManager::UploadConstants(const void* data, size_t size) {
    const auto stride = m_rhi.GetUniformBufferStride(size);
    auto& arena = m_uniforms.at(m_nFrameIndex);
    if (m_uniformCursor > arena.size || stride > arena.size - m_uniformCursor) {
        throw std::overflow_error("Vulkan frame uniform arena exhausted");
    }
    const auto offset = m_uniformCursor;
    m_rhi.UpdateUniformBuffer(arena, data, size, offset);
    m_uniformCursor += static_cast<uint32_t>(stride);
    return offset;
}

void VulkanGraphicsManager::BeginFrame(Frame& frame) {
    if (!IsInitialized() || m_frameRecording || m_computeActive ||
        frame.frameIndex != m_nFrameIndex || frame.batchContexts.size() > GfxConfiguration::kMaxSceneObjectCount ||
        frame.frameContext.numLights < 0 || frame.frameContext.numLights > MAX_LIGHTS) {
        throw std::logic_error("Invalid Vulkan graphics frame state");
    }
    if (m_rhi.HasSurface()) {
        if (!m_rhi.BeginFrame()) return;
    } else {
        m_commands = m_rhi.BeginCommands();
    }
    m_frameRecording = true;
    m_renderResources.at(m_nFrameIndex).Clear();
    const auto required = m_rhi.GetUniformBufferStride(sizeof(PerFrameConstants)) +
        m_rhi.GetUniformBufferStride(sizeof(LightInfo)) +
        m_rhi.GetUniformBufferStride(sizeof(PerBatchConstants)) * (frame.batchContexts.size() + 1) +
        m_rhi.GetUniformBufferStride(sizeof(ShadowMapConstants)) * (MAX_LIGHTS * 6 + 1);
    if (required > (std::numeric_limits<uint32_t>::max)()) {
        throw std::overflow_error("Vulkan dynamic uniform offsets exceed their 32-bit range");
    }
    if (m_uniforms.empty() || m_uniforms.front().size < required) {
        m_rhi.WaitGPU();
        m_rhi.DestroyUniformBuffers(m_uniforms);
        m_rhi.createUniformBuffers(m_uniforms, required);
    }
    m_uniformCursor = 0;
    m_frameOffset = UploadConstants(static_cast<const PerFrameConstants*>(&frame.frameContext),
                                     sizeof(PerFrameConstants));
    m_lightOffset = UploadConstants(&frame.lightInfo, sizeof(LightInfo));
    m_batchOffsets.resize(frame.batchContexts.size());
    for (const auto& batch : frame.batchContexts) {
        if (!batch || batch->batchIndex < 0 || static_cast<size_t>(batch->batchIndex) >= m_batchOffsets.size()) {
            throw std::invalid_argument("Invalid Vulkan draw batch index");
        }
        m_batchOffsets[batch->batchIndex] =
            UploadConstants(static_cast<const PerBatchConstants*>(batch.get()), sizeof(PerBatchConstants));
    }
    PerBatchConstants identity{};
    BuildIdentityMatrix(identity.modelMatrix);
    m_skyOffset = UploadConstants(&identity, sizeof(identity));
    ShadowMapConstants empty{};
    m_passOffset = UploadConstants(&empty, sizeof(empty));
}

void VulkanGraphicsManager::EndFrame(Frame&) {
    if (!m_frameRecording) return;
    if (m_rendering) throw std::logic_error("Vulkan frame ended inside a render target");
    if (m_commands) {
        m_commands->Submit();
        m_commands.reset();
    } else {
        m_rhi.EndFrame();
    }
    m_frameRecording = false;
    m_nFrameIndex = (m_nFrameIndex + 1) % GfxConfiguration::kMaxInFlightFrameCount;
}

void VulkanGraphicsManager::Present() {}

void VulkanGraphicsManager::Draw() {
    if (m_frameRecording) GraphicsManager::Draw();
}

void VulkanGraphicsManager::BeginRenderTarget(VulkanRHI::Texture* color,
    VulkanRHI::Texture& depth, bool clear, uint32_t firstLayer, uint32_t layers) {
    if (m_rendering || (color && (color->width != depth.width || color->height != depth.height ||
                                  color->samples != depth.samples)) ||
        firstLayer >= depth.layers || !layers || layers > depth.layers - firstLayer) {
        throw std::invalid_argument("Invalid Vulkan render target or layer range");
    }
    auto command = Command();
    auto& resources = m_renderResources.at(m_nFrameIndex);
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::ImageView> views;
    vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    if (color) {
        m_rhi.TransitionTexture(command, *color, Whole(*color), vk::ImageLayout::eColorAttachmentOptimal);
        attachments.emplace_back(vk::AttachmentDescriptionFlags{}, color->format, color->samples,
            clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
        views.push_back(color->descriptor);
    }
    const vk::ImageSubresourceRange depthRange(depth.aspect, 0, 1, firstLayer, layers);
    m_rhi.TransitionTexture(command, depth, depthRange, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    attachments.emplace_back(vk::AttachmentDescriptionFlags{}, depth.format, depth.samples,
        clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    resources.views.push_back(m_rhi.GetDevice().createImageViewUnique(
        {{}, depth.image, layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray,
         depth.format, {}, depthRange}));
    views.push_back(*resources.views.back());
    const vk::AttachmentReference depthReference(color ? 1u : 0u,
                                                  vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::SubpassDescription subpass;
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    if (color) subpass.setColorAttachments(colorReference);
    subpass.setPDepthStencilAttachment(&depthReference);
    const auto stages = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
    const auto accesses = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite |
                          vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    const vk::SubpassDependency dependency(VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eAllCommands, stages,
        vk::AccessFlagBits::eMemoryWrite, accesses);
    resources.passes.push_back(m_rhi.GetDevice().createRenderPassUnique({{}, attachments, subpass, dependency}));
    m_renderPass = *resources.passes.back();
    resources.framebuffers.push_back(m_rhi.GetDevice().createFramebufferUnique(
        {{}, m_renderPass, views, depth.width, depth.height, layers}));
    std::vector<vk::ClearValue> values;
    if (color) values.emplace_back(vk::ClearColorValue(std::array<float, 4>{
        m_Frames.at(m_nFrameIndex).clearColor[0], m_Frames.at(m_nFrameIndex).clearColor[1],
        m_Frames.at(m_nFrameIndex).clearColor[2], 1.0f}));
    values.emplace_back(vk::ClearDepthStencilValue(1.0f, 0));
    command.beginRenderPass({m_renderPass, *resources.framebuffers.back(),
        {{0, 0}, {depth.width, depth.height}}, values}, vk::SubpassContents::eInline);
    // Shadow projection and cube sampling share the existing positive-Y texture convention.
    command.setViewport(0, color ? vk::Viewport(0, float(depth.height), float(depth.width), -float(depth.height), 0, 1)
                                 : vk::Viewport(0, 0, float(depth.width), float(depth.height), 0, 1));
    command.setScissor(0, vk::Rect2D({0, 0}, {depth.width, depth.height}));
    m_renderColor = color;
    m_renderDepth = &depth;
    m_rendering = true;
    m_graphicsPipeline.reset();
}

void VulkanGraphicsManager::FinishRenderTarget() {
    if (!m_rendering) throw std::logic_error("No Vulkan render target is active");
    Command().endRenderPass();
    m_rendering = false;
    m_graphicsPipeline.reset();
}

void VulkanGraphicsManager::BeginPass(Frame& frame) {
    if (!m_frameRecording) throw std::logic_error("Vulkan pass requires a recording frame");
    if (frame.renderToTexture) {
        if (frame.colorTextures.size() != (frame.enableMSAA ? 2u : 1u)) {
            throw std::invalid_argument("Vulkan offscreen targets are incomplete");
        }
        BeginRenderTarget(&GetTextureResource(frame.colorTextures[frame.enableMSAA ? 1 : 0]),
                           GetTextureResource(frame.depthTexture), frame.clearRT);
    }
}

void VulkanGraphicsManager::EndPass(Frame& frame) {
    if (!frame.renderToTexture) return;
    FinishRenderTarget();
    if (!frame.enableMSAA) {
        auto& color = GetTextureResource(frame.colorTextures[0]);
        m_rhi.TransitionTexture(Command(), color, Whole(color), vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

void VulkanGraphicsManager::BindDraw(const Frame& frame, const material_textures& material, uint32_t batchOffset) {
    if (!m_rendering || !m_graphicsPipeline || !m_graphicsPipeline->pipelineState) {
        throw std::logic_error("Vulkan draw requires a live graphics pipeline and target");
    }
    auto& resources = m_renderResources.at(m_nFrameIndex);
    constexpr uint32_t block = 64;
    const auto device = m_rhi.GetDevice();
    if (resources.descriptorCount % block == 0) {
        const std::array sizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 4 * block),
            vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 12 * block),
            vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 2 * block),
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, block)};
        resources.descriptorPools.push_back(device.createDescriptorPoolUnique({{}, block, sizes}));
    }
    const auto set = device.allocateDescriptorSets(
        {*resources.descriptorPools.back(), *m_graphicsPipeline->descriptorSetLayout}).front();
    ++resources.descriptorCount;
    const auto buffer = m_uniforms.at(m_nFrameIndex).buffer;
    const std::array bufferInfos{
        vk::DescriptorBufferInfo(buffer, 0, sizeof(PerFrameConstants)),
        vk::DescriptorBufferInfo(buffer, 0, sizeof(PerBatchConstants)),
        vk::DescriptorBufferInfo(buffer, 0, sizeof(LightInfo)),
        vk::DescriptorBufferInfo(buffer, 0, sizeof(ShadowMapConstants))};
    const std::array<const TextureBase*, 12> textures{
        &material.diffuseMap, &material.normalMap, &material.metallicMap,
        &material.roughnessMap, &material.aoMap, &material.heightMap,
        &frame.brdfLUT, &frame.frameContext.shadowMap, &frame.frameContext.globalShadowMap,
        &frame.frameContext.cubeShadowMap, &frame.skybox, &frame.terrainHeightMap};
    std::array<vk::DescriptorImageInfo, 12> images;
    std::vector<vk::WriteDescriptorSet> writes;
    for (uint32_t index = 0; index < bufferInfos.size(); ++index) {
        writes.emplace_back(set, VulkanShaderBindings::kPerFrame + index, 0, 1,
                             vk::DescriptorType::eUniformBufferDynamic, nullptr, &bufferInfos[index]);
    }
    for (uint32_t index = 0; index < textures.size(); ++index) {
        const TextureBase* selected = textures[index];
        if (!selected->handler || (!m_renderColor && index >= 7 && index <= 10)) {
            if (index == 7 || index == 8) selected = &m_fallbackShadow;
            else if (index == 9) selected = &m_fallbackCubeShadow;
            else if (index == 10) selected = &m_fallbackSky;
            else selected = &m_fallback2D[index == 1 ? 1 : index == 2 ? 2 : 0];
        }
        const auto& texture = GetTextureResource(*selected);
        images[index] = {{}, texture.descriptor, texture.aspect == vk::ImageAspectFlagBits::eColor
            ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eDepthStencilReadOnlyOptimal};
        writes.emplace_back(set, VulkanShaderBindings::kTextureBase + index, 0, 1,
                             vk::DescriptorType::eSampledImage, &images[index]);
    }
    const vk::DescriptorImageInfo sampler(*m_sampler, {}, {});
    writes.emplace_back(set, VulkanShaderBindings::kSamplerBase, 0, 1, vk::DescriptorType::eSampler, &sampler);
    const vk::DescriptorImageInfo shadowSampler(*m_shadowSampler, {}, {});
    writes.emplace_back(set, VulkanShaderBindings::kShadowSampler, 0, 1,
                         vk::DescriptorType::eSampler, &shadowSampler);
    device.updateDescriptorSets(writes, {});
    const std::array offsets{m_frameOffset, batchOffset, m_lightOffset, m_passOffset};
    Command().bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *m_graphicsPipeline->pipelineLayout, 0, set, offsets);
}

void VulkanGraphicsManager::DrawBatches(const Frame& frame) {
    for (const auto& context : frame.batchContexts) {
        const auto* batch = dynamic_cast<const VulkanDrawBatchContext*>(context.get());
        if (!batch || !batch->node) throw std::invalid_argument("Foreign Vulkan draw batch");
        if (!batch->node->Visible() || (!m_renderColor && !batch->node->CastShadow())) continue;
        BindDraw(frame, batch->material, m_batchOffsets.at(batch->batchIndex));
        const auto count = m_graphicsPipeline->a2vType == A2V_TYPES_POS_ONLY ? 1u : 4u;
        const std::array<vk::DeviceSize, 4> offsets{};
        Command().bindVertexBuffers(0, {count, batch->vertices.data()}, {count, offsets.data()});
        Command().bindIndexBuffer(batch->indices, 0, batch->indexType);
        Command().drawIndexed(batch->indexCount, 1, 0, 0, 0);
    }
}

void VulkanGraphicsManager::DrawBatch(const Frame& frame) {
    if (!m_shadowSingleFace) {
        DrawBatches(frame);
        return;
    }
    const auto pipeline = m_graphicsPipeline;
    for (uint32_t face = 0; face < 6; ++face) {
        if (face) {
            FinishRenderTarget();
            BeginRenderTarget(nullptr, *m_renderDepth, false, m_shadowFirstLayer + face, 1);
            SetPipelineState(pipeline, frame);
        }
        auto constants = m_shadowConstants;
        constants.shadowMatrices[0] = m_shadowConstants.shadowMatrices[face];
        m_passOffset = UploadConstants(&constants, sizeof(constants));
        DrawBatches(frame);
    }
}

void VulkanGraphicsManager::MSAAResolve(std::optional<std::reference_wrapper<Texture2D>> target,
                                        Texture2D& source) {
    if (!target || m_rendering) throw std::invalid_argument("Vulkan resolve requires an offscreen target outside a pass");
    m_rhi.ResolveTexture(Command(), GetTextureResource(source), GetTextureResource(target->get()));
}

std::vector<uint8_t> VulkanGraphicsManager::ReadTexture(const TextureBase& texture, uint32_t layer, uint32_t mip) {
    if (m_frameRecording) throw std::logic_error("Vulkan readback requires a completed graphics frame");
    m_rhi.WaitGPU();
    return m_rhi.ReadTexture(GetTextureResource(texture), mip, layer);
}
