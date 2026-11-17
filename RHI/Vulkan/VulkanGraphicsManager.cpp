#include "VulkanGraphicsManager.hpp"
#include "VulkanShaderBindings.hpp"

#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

using namespace My;

VulkanGraphicsManager::~VulkanGraphicsManager() {
    m_commands.reset();
    try {
        m_rhi.AbortFrame();
        m_rhi.WaitGPU();
    } catch (const std::runtime_error& error) {
        std::cerr << "Vulkan graphics shutdown wait failed: " << error.what() << '\n';
    }
    m_rhi.DestroyUniformBuffers(m_uniforms);
}

int VulkanGraphicsManager::Initialize() {
    if (!m_rhi.GetDevice()) {
        throw std::logic_error("Vulkan graphics initialization requires an initialized RHI");
    }
    const auto result = GraphicsManager::Initialize();
    if (result != 0) return result;
    struct Rollback {
        VulkanGraphicsManager& owner;
        bool active = true;
        ~Rollback() { if (active) owner.Finalize(); }
    } rollback{*this};
    if (m_rhi.GetPhysicalDevice().getProperties().limits.maxUniformBufferRange < sizeof(LightInfo)) {
        throw std::runtime_error("Vulkan renderer requires the shared 28800-byte light uniform range");
    }
    InitializeFallbacks();
    rollback.active = false;
    return 0;
}

void VulkanGraphicsManager::Finalize() {
    m_InitPasses.clear();
    m_DispatchPasses.clear();
    m_DrawPasses.clear();
    GraphicsManager::Finalize();
    m_ownedTextures.clear();
    m_fallback2D = {};
    m_fallbackShadow = {};
    m_fallbackCubeShadow = {};
    m_fallbackSky = {};
    m_sampler.reset();
    m_shadowSampler.reset();
    m_nSceneRevision = 0;
    m_nFrameIndex = 0;
}

void VulkanGraphicsManager::EndScene() {
    m_commands.reset();
    m_rhi.AbortFrame();
    m_frameRecording = m_rendering = false;
    m_rhi.WaitGPU();
    for (auto& resources : m_renderResources) resources.Clear();
    m_graphicsPipeline.reset();
    m_rhi.DestroyUniformBuffers(m_uniforms);
    m_buffers.clear();
    m_materialTextures.clear();
    m_skyBatch = {};
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

void VulkanGraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    auto state = std::dynamic_pointer_cast<VulkanPipelineState>(pipelineState);
    if (state && state->pipelineType == PIPELINE_TYPE::GRAPHIC) {
        if (!m_frameRecording || !m_rendering || !m_pipelines ||
            !state->pipelineLayout || !state->descriptorSetLayout) {
            throw std::logic_error("Vulkan graphics binding requires an active target and pipeline manager");
        }
        if (state->sampleCount != static_cast<uint32_t>(m_renderDepth->samples) ||
            (state->pixelFormat == PIXEL_FORMAT::UNKNOWN) != (m_renderColor == nullptr)) {
            throw std::invalid_argument("Vulkan pipeline does not match the target attachment/sample configuration");
        }
        if (!state->pipelineState) m_pipelines->CreateGraphicsPipeline(*state, m_renderPass);
        m_graphicsPipeline = std::move(state);
        Command().bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline->pipelineState);
        return;
    }
    if (!m_computeActive || !state || state->pipelineType != PIPELINE_TYPE::COMPUTE ||
        !state->pipelineState || !state->pipelineLayout || !state->descriptorSetLayout) {
        throw std::logic_error("Vulkan compute binding requires a live compute pipeline and pass");
    }
    if (state->computeShaderName != "integrateBRDF.comp") {
        throw std::invalid_argument("Vulkan graphics compute binding currently supports the shared BRDF kernel only");
    }
    m_computePipeline = std::move(state);
}

void VulkanGraphicsManager::GenerateCubemapArray(TextureCubeArray& texture_array) {
    if (texture_array.size > (std::numeric_limits<uint32_t>::max)() / 6) {
        throw std::invalid_argument("Vulkan cube array layer count overflows");
    }
    if (texture_array.pixel_format != PIXEL_FORMAT::D32 || texture_array.samples != 1) {
        throw std::invalid_argument("Vulkan cube-array generation currently requires D32");
    }
    GenerateTextureResource(texture_array, texture_array.size * 6,
        vk::ImageViewType::eCubeArray,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::Format::eD32Sfloat);
    InitializeDepth(texture_array);
}

void VulkanGraphicsManager::GenerateTextureArray(Texture2DArray& texture_array) {
    if (texture_array.pixel_format != PIXEL_FORMAT::D32 || texture_array.samples != 1) {
        throw std::invalid_argument("Vulkan shadow-array generation requires D32");
    }
    GenerateTextureResource(texture_array, texture_array.size,
        vk::ImageViewType::e2DArray,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::Format::eD32Sfloat);
    InitializeDepth(texture_array);
}

void VulkanGraphicsManager::ReleaseTexture(TextureBase& texture) {
    if (!texture.handler && !texture.buff) return;
    if (m_frameRecording) throw std::logic_error("Cannot release a texture during Vulkan frame recording");
    GetTextureResource(texture);
    m_rhi.WaitGPU();
    for (auto& resources : m_renderResources) resources.Clear();
    if (m_computeOutput == texture.handler) m_computeOutput = 0;
    m_ownedTextures.erase(texture.handler);
    texture.handler = 0;
    texture.buff = 0;
}

void VulkanGraphicsManager::GenerateTextureResource(
    TextureBase& texture, uint32_t layers, vk::ImageViewType viewType,
    vk::ImageUsageFlags usage, vk::Format format) {
    if (texture.handler || texture.buff ||
        m_nextTexture == (std::numeric_limits<TextureHandler>::max)()) {
        throw std::invalid_argument("Vulkan texture generation requires an empty descriptor");
    }
    auto owned = std::make_unique<OwnedTexture>(m_rhi);
    owned->texture = m_rhi.CreateTexture(texture.width, texture.height, format,
        texture.mips, layers, viewType, usage,
        static_cast<vk::SampleCountFlagBits>(texture.samples));
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
         texture.pixel_format != PIXEL_FORMAT::RGBA32) || texture.mips != 1 || texture.samples != 1) {
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
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 2),
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
    return ReadTexture(texture, 0, 0);
}

void VulkanGraphicsManager::BeginCompute() {
    if (!IsInitialized() || m_computeActive || m_frameRecording) {
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
