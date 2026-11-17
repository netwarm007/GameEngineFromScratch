#include "VulkanGraphicsManager.hpp"

#include <stdexcept>

#include "PointShadowMatrices.hpp"

using namespace My;

void VulkanGraphicsManager::BeginShadowMap(int32_t lightIndex, const TextureBase* descriptor,
                                            int32_t layerIndex, const Frame& frame) {
    if (!descriptor || lightIndex < 0 || lightIndex >= frame.frameContext.numLights ||
        layerIndex < 0 || m_rendering) {
        throw std::invalid_argument("Invalid Vulkan shadow pass input");
    }
    const auto& light = frame.lightInfo.lights[lightIndex];
    if (!light.lightCastShadow || layerIndex != light.lightShadowMapIndex) {
        throw std::invalid_argument("Vulkan shadow pass must consume the prepared light layer");
    }
    auto& texture = GetTextureResource(*descriptor);
    m_shadowOmni = light.lightType == LightType::Omni;
    m_shadowSingleFace = false;
    if (m_shadowOmni) {
        if (!m_pipelines) throw std::logic_error("Vulkan shadows require a pipeline manager");
        const auto pipeline = std::dynamic_pointer_cast<VulkanPipelineState>(
            m_pipelines->GetPipelineState("Omni Light Shadow Map"));
        if (!pipeline) throw std::logic_error("Vulkan omni shadow pipeline is unavailable");
        m_shadowSingleFace = pipeline->vertexShaderName == "shadowmap_omni_single.vert";
    }
    const uint64_t first = uint64_t{static_cast<uint32_t>(layerIndex)} * (m_shadowOmni ? 6 : 1);
    const uint32_t count = m_shadowOmni ? 6 : 1;
    if (first >= texture.layers || count > texture.layers - first) {
        throw std::invalid_argument("Vulkan shadow layer exceeds its allocated texture");
    }
    m_shadowFirstLayer = static_cast<uint32_t>(first);
    const vk::ImageSubresourceRange range(texture.aspect, 0, 1, m_shadowFirstLayer, count);
    auto command = Command();
    m_rhi.TransitionTexture(command, texture, range, vk::ImageLayout::eTransferDstOptimal);
    command.clearDepthStencilImage(texture.image, vk::ImageLayout::eTransferDstOptimal,
                                    vk::ClearDepthStencilValue(1.0f, 0), range);
    m_shadowConstants = {};
    m_shadowConstants.light_index = lightIndex;
    m_shadowConstants.shadowmap_layer_index = static_cast<float>(layerIndex);
    m_shadowConstants.near_plane = 1.0f;
    m_shadowConstants.far_plane = POINT_SHADOW_FAR_PLANE;
    if (m_shadowOmni) {
        const Vector3f position{light.lightPosition[0], light.lightPosition[1], light.lightPosition[2]};
        BuildPointShadowMatrices(m_shadowConstants.shadowMatrices, position, light.lightProjectionMatrix);
    }
    m_passOffset = UploadConstants(&m_shadowConstants, sizeof(m_shadowConstants));
    BeginRenderTarget(nullptr, texture, false,
        m_shadowOmni && !m_shadowSingleFace ? 0u : m_shadowFirstLayer,
        m_shadowOmni && !m_shadowSingleFace ? texture.layers : 1u);
}

void VulkanGraphicsManager::EndShadowMap(const TextureBase* descriptor, int32_t, const Frame&) {
    if (!descriptor || m_renderDepth != &GetTextureResource(*descriptor)) {
        throw std::invalid_argument("Mismatched Vulkan shadow target");
    }
    FinishRenderTarget();
    auto& texture = GetTextureResource(*descriptor);
    m_rhi.TransitionTexture(Command(), texture, {texture.aspect, 0, 1, 0, texture.layers},
                            vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    m_shadowOmni = m_shadowSingleFace = false;
}

void VulkanGraphicsManager::SetShadowMaps(const Frame& frame) {
    if (!m_rendering || !m_graphicsPipeline) {
        throw std::logic_error("Vulkan shadow sampling requires a bound graphics pipeline");
    }
    for (const TextureBase* texture : {static_cast<const TextureBase*>(&frame.frameContext.shadowMap),
                                       static_cast<const TextureBase*>(&frame.frameContext.globalShadowMap),
                                       static_cast<const TextureBase*>(&frame.frameContext.cubeShadowMap)}) {
        if (texture->handler) GetTextureResource(*texture);
    }
    // Per-draw descriptors bind these shared arrays alongside material textures.
}

void VulkanGraphicsManager::initializeSkyBox(const Scene& scene) {
    if (!scene.SkyBox) throw std::invalid_argument("Vulkan skybox initialization requires an environment");
    auto& first = scene.SkyBox->GetTexture(0);
    auto& radiance = scene.SkyBox->GetTexture(12);
    const auto firstImage = first.GetTextureImage();
    const auto radianceImage = radiance.GetTextureImage();
    if (!firstImage || !radianceImage) throw std::runtime_error("Vulkan environment images were not loaded");
    OwnedTexture firstUpload(m_rhi);
    firstUpload.texture = m_rhi.createTextureImage(*firstImage);
    auto& sky = m_Frames.front().skybox;
    sky.width = firstImage->Width;
    sky.height = firstImage->Height;
    sky.size = 2;
    sky.mips = static_cast<uint32_t>((std::max)(radianceImage->mipmaps.size(), size_t{2}));
    sky.pixel_format = firstImage->pixel_format;
    GenerateTextureResource(sky, 12, vk::ImageViewType::eCubeArray,
        vk::ImageUsageFlagBits::eSampled, firstUpload.texture.format);
    auto& target = GetTextureResource(sky);
    for (uint32_t index = 0; index < 18; ++index) {
        OwnedTexture upload(m_rhi);
        const auto image = scene.SkyBox->GetTexture(index).GetTextureImage();
        if (!image) throw std::runtime_error("Vulkan environment face was not loaded");
        upload.texture = m_rhi.createTextureImage(*image);
        const auto levels = index < 6 ? target.mipLevels : index < 12 ? 1u : upload.texture.mipLevels;
        if (upload.texture.format != target.format ||
            (index >= 12 && levels != target.mipLevels)) {
            throw std::invalid_argument("Incompatible Vulkan environment face format or radiance mip chain");
        }
        m_rhi.ExecuteCommands([&](vk::CommandBuffer command) {
            for (uint32_t level = 0; level < levels; ++level) {
                const auto mip = index >= 6 && index < 12 ? 1u : level;
                const auto layer = index < 12 ? index % 6 : index % 6 + 6;
                const uint32_t width = (std::max)(1u, target.width >> mip);
                const uint32_t height = (std::max)(1u, target.height >> mip);
                const vk::ImageSubresourceRange targetRange(target.aspect, mip, 1, layer, 1);
                if (index < 6 && level >= upload.texture.mipLevels) {
                    const auto features = m_rhi.GetPhysicalDevice().getFormatProperties(target.format).optimalTilingFeatures;
                    const auto required = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;
                    if ((features & required) != required) {
                        throw std::invalid_argument("Vulkan visible environment requires a full mip chain or blit support");
                    }
                    const vk::ImageSubresourceRange previous(target.aspect, mip - 1, 1, layer, 1);
                    m_rhi.TransitionTexture(command, target, previous, vk::ImageLayout::eTransferSrcOptimal);
                    m_rhi.TransitionTexture(command, target, targetRange, vk::ImageLayout::eTransferDstOptimal);
                    vk::ImageBlit blit;
                    blit.srcSubresource = {target.aspect, mip - 1, layer, 1};
                    blit.srcOffsets[1] = vk::Offset3D((std::max)(1u, target.width >> (mip - 1)),
                                                     (std::max)(1u, target.height >> (mip - 1)), 1);
                    blit.dstSubresource = {target.aspect, mip, layer, 1};
                    blit.dstOffsets[1] = vk::Offset3D(width, height, 1);
                    command.blitImage(target.image, vk::ImageLayout::eTransferSrcOptimal,
                        target.image, vk::ImageLayout::eTransferDstOptimal, blit,
                        features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear ? vk::Filter::eLinear : vk::Filter::eNearest);
                    m_rhi.TransitionTexture(command, target, previous, vk::ImageLayout::eShaderReadOnlyOptimal);
                    m_rhi.TransitionTexture(command, target, targetRange, vk::ImageLayout::eShaderReadOnlyOptimal);
                    continue;
                }
                if ((std::max)(1u, upload.texture.width >> level) != width ||
                    (std::max)(1u, upload.texture.height >> level) != height) {
                    throw std::invalid_argument("Vulkan environment packing dimensions do not match shared shader LODs");
                }
                const vk::ImageSubresourceRange sourceRange(upload.texture.aspect, level, 1, 0, 1);
                m_rhi.TransitionTexture(command, upload.texture, sourceRange, vk::ImageLayout::eTransferSrcOptimal);
                m_rhi.TransitionTexture(command, target, targetRange, vk::ImageLayout::eTransferDstOptimal);
                command.copyImage(upload.texture.image, vk::ImageLayout::eTransferSrcOptimal,
                    target.image, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageCopy({target.aspect, level, 0, 1}, {0, 0, 0},
                                  {target.aspect, mip, layer, 1}, {0, 0, 0}, {width, height, 1}));
                m_rhi.TransitionTexture(command, target, targetRange, vk::ImageLayout::eShaderReadOnlyOptimal);
            }
        });
    }
    m_rhi.ExecuteCommands([&](vk::CommandBuffer command) {
        m_rhi.TransitionTexture(command, target, {target.aspect, 0, target.mipLevels, 0, target.layers},
                                vk::ImageLayout::eShaderReadOnlyOptimal);
    });
    m_skyBatch.vertices.fill(UploadVertex(SceneObjectSkyBox::skyboxVertices.data(),
                                          sizeof(SceneObjectSkyBox::skyboxVertices)));
    m_skyBatch.indices = UploadIndex(SceneObjectSkyBox::skyboxIndices.data(),
                                     sizeof(SceneObjectSkyBox::skyboxIndices));
    m_skyBatch.indexType = vk::IndexType::eUint16;
    m_skyBatch.indexCount = static_cast<uint32_t>(SceneObjectSkyBox::skyboxIndices.size());
}

void VulkanGraphicsManager::DrawSkyBox(const Frame& frame) {
    if (!frame.skybox.handler) return;
    if (!m_skyBatch.indices) throw std::logic_error("Vulkan skybox geometry is unavailable");
    BindDraw(frame, {}, m_skyOffset);
    const std::array<vk::DeviceSize, 2> offsets{};
    Command().bindVertexBuffers(0, {2, m_skyBatch.vertices.data()}, offsets);
    Command().bindIndexBuffer(m_skyBatch.indices, 0, m_skyBatch.indexType);
    Command().drawIndexed(m_skyBatch.indexCount, 1, 0, 0, 0);
}
