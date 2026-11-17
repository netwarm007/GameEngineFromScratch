#include "VulkanPipelineStateManager.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include "VulkanShaderBindings.hpp"

using namespace My;

namespace {
void ValidateEntryPoint(const Buffer& code, vk::ShaderStageFlagBits stage, const std::string& entry) {
    const auto invalid = [&] { return std::invalid_argument("Invalid Vulkan shader entry point: " + entry); };
    if (!code.GetData() || code.GetDataSize() < 20 || code.GetDataSize() % 4) throw invalid();
    const auto word = [&](size_t index) {
        uint32_t value;
        std::memcpy(&value, code.GetData() + index * 4, sizeof(value));
        return value;
    };
    if (word(0) != 0x07230203u) throw invalid();
    uint32_t model;
    switch (stage) {
        case vk::ShaderStageFlagBits::eVertex: model = 0; break;
        case vk::ShaderStageFlagBits::eGeometry: model = 3; break;
        case vk::ShaderStageFlagBits::eFragment: model = 4; break;
        case vk::ShaderStageFlagBits::eCompute: model = 5; break;
        default: throw invalid();
    }
    bool found = false;
    for (size_t offset = 5; offset < code.GetDataSize() / 4;) {
        const auto instruction = word(offset);
        const auto count = instruction >> 16;
        if (!count || count > code.GetDataSize() / 4 - offset) throw invalid();
        if ((instruction & 0xffffu) == 17) {  // OpCapability
            if (count != 2) throw invalid();
            const auto capability = word(offset + 1);
            // Device setup does not yet enable ShaderLayer/ShaderViewportIndex/EXT layer output.
            if (capability == 69 || capability == 70 || capability == 5254) {
                throw std::runtime_error("Vulkan vertex-layer/viewport output requires enabled device "
                                         "features or the shared single-face shadow adaptation");
            }
        }
        if ((instruction & 0xffffu) == 15) {  // OpEntryPoint
            if (count < 4) throw invalid();
            const auto* name = reinterpret_cast<const char*>(code.GetData() + (offset + 3) * 4);
            const auto* end = static_cast<const char*>(std::memchr(name, 0, (count - 3) * 4));
            if (!end) throw invalid();
            found |= word(offset + 1) == model &&
                     std::string_view(name, static_cast<size_t>(end - name)) == entry;
        }
        offset += count;
    }
    if (!found) throw invalid();
}

std::vector<vk::PipelineShaderStageCreateInfo> ShaderStages(const VulkanPipelineState& state) {
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    for (const auto& shader : state.stages) {
        stages.emplace_back(vk::PipelineShaderStageCreateFlags{}, shader.stage,
                            *shader.module, shader.entry.c_str());
    }
    return stages;
}

vk::CompareOp DepthCompare(DEPTH_TEST_MODE mode) {
    switch (mode) {
        case DEPTH_TEST_MODE::NONE:
        case DEPTH_TEST_MODE::ALWAYS: return vk::CompareOp::eAlways;
        case DEPTH_TEST_MODE::LARGE: return vk::CompareOp::eGreater;
        case DEPTH_TEST_MODE::LARGE_EQUAL: return vk::CompareOp::eGreaterOrEqual;
        case DEPTH_TEST_MODE::EQUAL: return vk::CompareOp::eEqual;
        case DEPTH_TEST_MODE::LESS_EQUAL: return vk::CompareOp::eLessOrEqual;
        case DEPTH_TEST_MODE::LESS: return vk::CompareOp::eLess;
        case DEPTH_TEST_MODE::NOT_EQUAL: return vk::CompareOp::eNotEqual;
        case DEPTH_TEST_MODE::NEVER: return vk::CompareOp::eNever;
        default: throw std::invalid_argument("Unsupported Vulkan depth comparison");
    }
}
}  // namespace

int VulkanPipelineStateManager::Initialize() {
    if (!m_pApp || !m_pipelineStates.empty()) {
        throw std::logic_error("Vulkan registry initialization requires an application and empty registry");
    }
    struct Rollback {
        VulkanPipelineStateManager& manager;
        bool active = true;
        ~Rollback() { if (active) manager.Clear(); }
    } rollback{*this};
    const auto result = PipelineStateManager::Initialize();
    if (result == 0) rollback.active = false;
    return result;
}

bool VulkanPipelineStateManager::InitializePipelineState(
    PipelineState** ppPipelineState) {
    if (!ppPipelineState || !*ppPipelineState || !m_rhi.GetDevice()) {
        throw std::logic_error("Vulkan pipeline creation requires a definition and initialized device");
    }
    auto definition = **ppPipelineState;
    if (definition.pixelFormat == PIXEL_FORMAT::UNKNOWN &&
        definition.pixelShaderName == "shadowmap.frag") {
        definition.pixelShaderName.clear();
    }
    const auto features = m_rhi.GetPhysicalDevice().getFeatures();
    if (definition.pipelineStateName == "Omni Light Shadow Map" &&
        (!features.geometryShader || definition.vertexShaderName == "shadowmap_omni_metal.vert")) {
        definition.vertexShaderName = "shadowmap_omni_single.vert";
        definition.geometryShaderName.clear();
    }
    if (definition.pipelineStateName.empty() || m_pipelineStates.contains(definition.pipelineStateName)) {
        throw std::invalid_argument("Empty or duplicate Vulkan pipeline name");
    }
    if (!definition.tessControlShaderName.empty() || !definition.tessEvaluateShaderName.empty() ||
        !definition.meshShaderName.empty()) {
        throw std::invalid_argument("Vulkan tessellation/mesh pipelines are not implemented");
    }
    if (definition.pipelineType != PIPELINE_TYPE::COMPUTE &&
        definition.pipelineType != PIPELINE_TYPE::GRAPHIC) {
        throw std::invalid_argument("Unsupported Vulkan pipeline type");
    }
    const bool compute = definition.pipelineType == PIPELINE_TYPE::COMPUTE;
    if (compute ? definition.computeShaderName.empty() || !definition.vertexShaderName.empty() ||
                      !definition.pixelShaderName.empty() || !definition.geometryShaderName.empty()
                : definition.vertexShaderName.empty() || !definition.computeShaderName.empty()) {
        throw std::invalid_argument("Vulkan pipeline has missing or incompatible shader stages");
    }
    if (!definition.geometryShaderName.empty() && !features.geometryShader) {
        throw std::runtime_error("Vulkan pipeline requires geometry shader support");
    }
    auto state = std::make_unique<VulkanPipelineState>(definition);
    const auto load = [&](const std::string& name, const char* suffix, vk::ShaderStageFlagBits stage) {
        if (name.empty()) return;
        if (!name.ends_with(suffix) || name.find_first_of("/\\") != std::string::npos) {
            throw std::invalid_argument("Invalid Vulkan engine shader stage name: " + name);
        }
        const auto path = "Shaders/Vulkan/Engine/" + name + ".spv";
        const auto code = m_assets.SyncOpenAndReadBinary(path.c_str());
        if (!code.GetDataSize()) throw std::runtime_error("Missing Vulkan engine shader: " + path);
        auto entry = name;
        std::replace(entry.begin(), entry.end(), '.', '_');
        entry += "_main";
        ValidateEntryPoint(code, stage, entry);
        vk::UniqueShaderModule module(m_rhi.createShaderModule(code), m_rhi.GetDevice());
        state->stages.push_back({std::move(module), stage, std::move(entry)});
    };
    load(definition.vertexShaderName, ".vert", vk::ShaderStageFlagBits::eVertex);
    load(definition.pixelShaderName, ".frag", vk::ShaderStageFlagBits::eFragment);
    load(definition.geometryShaderName, ".geom", vk::ShaderStageFlagBits::eGeometry);
    load(definition.computeShaderName, ".comp", vk::ShaderStageFlagBits::eCompute);

    using namespace VulkanShaderBindings;
    auto visibility = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
                      vk::ShaderStageFlagBits::eCompute;
    if (features.geometryShader) visibility |= vk::ShaderStageFlagBits::eGeometry;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (const auto binding : {kPerFrame, kPerBatch, kLightInfo, kPassConstants}) {
        bindings.emplace_back(binding, vk::DescriptorType::eUniformBufferDynamic, 1, visibility);
    }
    for (uint32_t texture = 0; texture < 12; ++texture) {
        bindings.emplace_back(kTextureBase + texture, vk::DescriptorType::eSampledImage, 1, visibility);
    }
    bindings.emplace_back(kSamplerBase, vk::DescriptorType::eSampler, 1, visibility);
    bindings.emplace_back(kShadowSampler, vk::DescriptorType::eSampler, 1, visibility);
    bindings.emplace_back(kStorageImageBase, vk::DescriptorType::eStorageImage, 1, visibility);
    const auto device = m_rhi.GetDevice();
    state->descriptorSetLayout = device.createDescriptorSetLayoutUnique({{}, bindings});
    const auto layout = *state->descriptorSetLayout;
    state->pipelineLayout = device.createPipelineLayoutUnique({{}, layout});
    if (compute) {
        const auto stages = ShaderStages(*state);
        state->pipelineState = device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo({}, stages.front(), *state->pipelineLayout)).value;
    }
    *ppPipelineState = state.release();
    return true;
}

std::shared_ptr<VulkanPipelineState> VulkanPipelineStateManager::CreateMaterialCopyPipeline(
    vk::RenderPass renderPass) {
    PipelineState definition;
    definition.pipelineStateName = "Material Texture Copy";
    definition.vertexShaderName = "passthrough.vert";
    definition.pixelShaderName = "texture.frag";
    definition.pipelineType = PIPELINE_TYPE::GRAPHIC;
    definition.a2vType = A2V_TYPES::A2V_TYPES_SIMPLE;
    definition.depthTestMode = DEPTH_TEST_MODE::NONE;
    definition.bDepthWrite = false;
    definition.stencilTestMode = STENCIL_TEST_MODE::NONE;
    definition.cullFaceMode = CULL_FACE_MODE::NONE;
    definition.pixelFormat = PIXEL_FORMAT::RGBA8;
    definition.sampleCount = 1;
    definition.flag = PIPELINE_FLAG::NONE;
    PipelineState* result = &definition;
    InitializePipelineState(&result);
    auto state = std::dynamic_pointer_cast<VulkanPipelineState>(std::shared_ptr<PipelineState>(result));
    CreateGraphicsPipeline(*state, renderPass);
    return state;
}

void VulkanPipelineStateManager::CreateGraphicsPipeline(VulkanPipelineState& state,
                                                         vk::RenderPass renderPass) {
    if (state.pipelineType != PIPELINE_TYPE::GRAPHIC || !renderPass ||
        !state.pipelineLayout || state.pipelineState) {
        throw std::logic_error("Vulkan graphics pipeline requires an unbuilt graphics definition and render pass");
    }
    vk::SampleCountFlagBits samples;
    switch (state.sampleCount) {
        case 1: samples = vk::SampleCountFlagBits::e1; break;
        case 2: samples = vk::SampleCountFlagBits::e2; break;
        case 4: samples = vk::SampleCountFlagBits::e4; break;
        case 8: samples = vk::SampleCountFlagBits::e8; break;
        case 16: samples = vk::SampleCountFlagBits::e16; break;
        case 32: samples = vk::SampleCountFlagBits::e32; break;
        case 64: samples = vk::SampleCountFlagBits::e64; break;
        default: throw std::invalid_argument("Invalid Vulkan pipeline sample count");
    }
    const auto limits = m_rhi.GetPhysicalDevice().getProperties().limits;
    if ((state.pixelFormat != PIXEL_FORMAT::UNKNOWN && !(limits.framebufferColorSampleCounts & samples)) ||
        (state.depthTestMode != DEPTH_TEST_MODE::NONE && !(limits.framebufferDepthSampleCounts & samples))) {
        throw std::runtime_error("Vulkan pipeline sample count is unsupported by this device");
    }
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
    const auto attribute = [&](uint32_t components) {
        const auto binding = static_cast<uint32_t>(bindings.size());
        bindings.emplace_back(binding, components * sizeof(float), vk::VertexInputRate::eVertex);
        attributes.emplace_back(binding, binding, components == 2 ? vk::Format::eR32G32Sfloat
                                                                  : vk::Format::eR32G32B32Sfloat, 0);
    };
    switch (state.a2vType) {
        case A2V_TYPES_FULL: attribute(3); attribute(3); attribute(2); attribute(3); break;
        case A2V_TYPES_SIMPLE: attribute(3); attribute(2); break;
        case A2V_TYPES_POS_ONLY: attribute(3); break;
        case A2V_TYPES_CUBE:
            attribute(3);
            if (state.vertexShaderName != "skybox.vert") attribute(3);
            break;
        case A2V_TYPES_NONE: break;
        default: throw std::invalid_argument("Unsupported Vulkan vertex input layout");
    }
    const vk::PipelineVertexInputStateCreateInfo vertex({}, bindings, attributes);
    const vk::PipelineInputAssemblyStateCreateInfo assembly({}, vk::PrimitiveTopology::eTriangleList);
    const vk::PipelineViewportStateCreateInfo viewport({}, 1, nullptr, 1, nullptr);
    vk::CullModeFlags cull;
    switch (state.cullFaceMode) {
        case CULL_FACE_MODE::NONE: cull = vk::CullModeFlagBits::eNone; break;
        case CULL_FACE_MODE::FRONT: cull = vk::CullModeFlagBits::eFront; break;
        case CULL_FACE_MODE::BACK: cull = vk::CullModeFlagBits::eBack; break;
        default: throw std::invalid_argument("Unsupported Vulkan cull mode");
    }
    // Paired with a negative-height viewport for the engine's Y-up rendering convention.
    const vk::PipelineRasterizationStateCreateInfo raster(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, cull,
        state.flag == PIPELINE_FLAG::SHADOW ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise,
        VK_FALSE, 0, 0, 0, 1);
    const vk::PipelineMultisampleStateCreateInfo multisample({}, samples);
    const vk::PipelineDepthStencilStateCreateInfo depth(
        {}, state.depthTestMode != DEPTH_TEST_MODE::NONE, state.bDepthWrite, DepthCompare(state.depthTestMode));
    const vk::ColorComponentFlags rgba = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                         vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendAttachmentState color;
    color.colorWriteMask = rgba;
    const vk::PipelineColorBlendStateCreateInfo blend(
        {}, VK_FALSE, vk::LogicOp::eCopy, state.pixelFormat == PIXEL_FORMAT::UNKNOWN ? 0u : 1u, &color);
    const std::array dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic({}, dynamicStates);
    const auto stages = ShaderStages(state);
    const vk::GraphicsPipelineCreateInfo info({}, stages, &vertex, &assembly, nullptr,
        &viewport, &raster, &multisample, &depth, &blend, &dynamic, *state.pipelineLayout, renderPass);
    state.pipelineState = m_rhi.GetDevice().createGraphicsPipelineUnique({}, info).value;
}

void VulkanPipelineStateManager::DestroyPipelineState(PipelineState& pipelineState) {
    auto& state = dynamic_cast<VulkanPipelineState&>(pipelineState);
    state.pipelineState.reset();
    state.stages.clear();
    state.pipelineLayout.reset();
    state.descriptorSetLayout.reset();
}

void VulkanPipelineStateManager::UnregisterPipelineState(PipelineState& pipelineState) {
    const auto found = m_pipelineStates.find(pipelineState.pipelineStateName);
    if (found == m_pipelineStates.end()) {
        throw std::invalid_argument("Vulkan pipeline is not registered: " + pipelineState.pipelineStateName);
    }
    const auto retained = found->second;
    m_pipelineStates.erase(found);
    DestroyPipelineState(*retained);
}
