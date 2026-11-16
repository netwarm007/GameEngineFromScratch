#include "Vulkan/VulkanPipelineStateManager.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "AssetLoader.hpp"
#include "BrdfReference.hpp"
#include "Vulkan/VulkanShaderBindings.hpp"

namespace {
using namespace My;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Resources {
    VulkanRHI& rhi;
    VulkanRHI::Texture source;
    VulkanRHI::Texture target;
    std::array<VulkanRHI::VertexBuffer, 2> vertices;
    ~Resources() {
        rhi.DestroyTexture(target);
        rhi.DestroyTexture(source);
        for (auto& vertex : vertices) {
            rhi.GetDevice().destroyBuffer(vertex.buffer);
            rhi.GetDevice().freeMemory(vertex.heap);
        }
    }
};

struct Descriptors {
    vk::UniqueDescriptorPool pool;
    vk::DescriptorSet set;
    Descriptors(vk::Device device, const VulkanPipelineState& state) {
        const std::array sizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 4),
            vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 12),
            vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1)};
        pool = device.createDescriptorPoolUnique({{}, 1, sizes});
        const auto layout = *state.descriptorSetLayout;
        set = device.allocateDescriptorSets({*pool, layout}).front();
    }
};

std::shared_ptr<VulkanPipelineState> Register(VulkanPipelineStateManager& manager,
                                             PipelineState definition) {
    Require(manager.RegisterPipelineState(definition), "Production pipeline registration failed");
    auto state = std::dynamic_pointer_cast<VulkanPipelineState>(
        manager.GetPipelineState(definition.pipelineStateName));
    Require(state && state->pipelineLayout && state->descriptorSetLayout,
            "Production pipeline has no descriptor/pipeline layout");
    return state;
}

PipelineState ComputeDefinition() {
    PipelineState definition;
    definition.pipelineStateName = "PBR BRDF CS";
    definition.pipelineType = PIPELINE_TYPE::COMPUTE;
    definition.computeShaderName = "integrateBRDF.comp";
    definition.flag = PIPELINE_FLAG::NONE;
    return definition;
}

class MismatchedAssets final : public AssetLoader {
   public:
    Buffer SyncOpenAndReadBinary(const char*) final {
        return AssetLoader::SyncOpenAndReadBinary("Shaders/Vulkan/Engine/raytrace.comp.spv");
    }
};

void CheckFailures(VulkanRHI& rhi, VulkanPipelineStateManager& manager) {
    auto definition = ComputeDefinition();
    definition.pipelineStateName = "Missing asset";
    definition.computeShaderName = "missing.comp";
    bool rejected = false;
    try {
        manager.RegisterPipelineState(definition);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    Require(rejected, "Missing production shader was accepted");
    definition.pipelineStateName = "Wrong stage";
    definition.computeShaderName = "basic.vert";
    rejected = false;
    try {
        manager.RegisterPipelineState(definition);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "A vertex shader was accepted as compute");
    MismatchedAssets mismatchedAssets;
    VulkanPipelineStateManager mismatchedManager(rhi, mismatchedAssets);
    rejected = false;
    try {
        auto expectedBrdf = ComputeDefinition();
        mismatchedManager.RegisterPipelineState(expectedBrdf);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "A binary with the wrong entry point was accepted");
    PipelineState layered;
    layered.pipelineStateName = "Unsupported vertex-layer feature";
    layered.vertexShaderName = "shadowmap_omni_metal.vert";
    layered.pixelShaderName = "shadowmap_omni.frag";
    layered.flag = PIPELINE_FLAG::SHADOW;
    rejected = false;
    try {
        manager.RegisterPipelineState(layered);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    Require(rejected, "Vertex-layer output was accepted without enabled device features");
    Buffer invalid(24);
    std::memset(invalid.GetData(), 0, invalid.GetDataSize());
    rejected = false;
    try {
        rhi.createShaderModule(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "Invalid SPIR-V reached the driver");
    rejected = false;
    try {
        rhi.ExecuteCommands([](vk::CommandBuffer) { throw std::runtime_error("Aborted recorder"); });
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    Require(rejected, "Immediate command recording failure was hidden");
}

std::shared_ptr<VulkanPipelineState> CheckCompute(VulkanRHI& rhi,
                                                  VulkanPipelineStateManager& manager) {
    const auto state = Register(manager, ComputeDefinition());
    Resources resources{rhi};
    resources.target = rhi.CreateTexture(512, 512, vk::Format::eR32G32B32A32Sfloat,
        1, 1, vk::ImageViewType::e2D, vk::ImageUsageFlagBits::eStorage);
    Descriptors descriptors(rhi.GetDevice(), *state);
    const vk::DescriptorImageInfo image({}, resources.target.descriptor, vk::ImageLayout::eGeneral);
    const vk::WriteDescriptorSet write(descriptors.set, VulkanShaderBindings::kStorageImageBase,
        0, 1, vk::DescriptorType::eStorageImage, &image);
    rhi.GetDevice().updateDescriptorSets(write, {});
    rhi.ExecuteCommands([&](vk::CommandBuffer command) {
        rhi.TransitionTexture(command, resources.target,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}, vk::ImageLayout::eGeneral);
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *state->pipelineState);
        const std::array<uint32_t, 4> offsets{};
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *state->pipelineLayout,
                                   0, descriptors.set, offsets);
        command.dispatch(512, 512, 1);
    });
    const auto bytes = rhi.ReadTexture(resources.target);
    Require(bytes.size() == 512 * 512 * 4 * sizeof(float), "Production BRDF readback size mismatch");
    for (const auto sample : {std::array<uint32_t, 2>{32, 64}, {255, 255}, {511, 511}}) {
        const auto reference = VulkanTest::ReferenceBrdf(sample[0], sample[1]);
        std::array<float, 4> pixel{};
        std::memcpy(pixel.data(), bytes.data() + (sample[1] * 512 + sample[0]) * sizeof(pixel),
                     sizeof(pixel));
        for (size_t channel = 0; channel < 4; ++channel) {
            if (!std::isfinite(pixel[channel]) ||
                std::abs(pixel[channel] - reference[channel % 2]) > 0.001) {
                throw std::runtime_error("Production BRDF channel " +
                    std::to_string(channel) + ": expected " + std::to_string(reference[channel % 2]) +
                    ", got " + std::to_string(pixel[channel]));
            }
        }
    }
    bool rejected = false;
    try {
        manager.RegisterPipelineState(*state);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected && manager.GetPipelineState("PBR BRDF CS") == state,
            "Duplicate registration replaced the live production pipeline");
    return state;
}

void CheckOffscreen(VulkanRHI& rhi, VulkanPipelineStateManager& manager) {
    const auto device = rhi.GetDevice();
    Resources resources{rhi};
    resources.source = rhi.CreateTexture(2, 2, vk::Format::eR8G8B8A8Unorm);
    resources.target = rhi.CreateTexture(16, 16, vk::Format::eR8G8B8A8Unorm,
        1, 1, vk::ImageViewType::e2D,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    const std::array<uint8_t, 16> colors{
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255};
    rhi.UpdateTexture(resources.source, colors.data(), colors.size());
    const std::array<float, 18> positions{
        -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0};
    const std::array<float, 12> uvs{0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0};
    rhi.createVertexBuffer(positions.data(), sizeof(positions), resources.vertices[0]);
    rhi.createVertexBuffer(uvs.data(), sizeof(uvs), resources.vertices[1]);
    const vk::AttachmentDescription attachment({}, resources.target.format, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    const vk::AttachmentReference color(0, vk::ImageLayout::eColorAttachmentOptimal);
    const vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, color);
    const auto renderPass = device.createRenderPassUnique({{}, attachment, subpass});
    const auto view = resources.target.descriptor;
    const auto framebuffer = device.createFramebufferUnique({{}, *renderPass, view, 16, 16, 1});
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = samplerInfo.minFilter = vk::Filter::eNearest;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW =
        vk::SamplerAddressMode::eClampToEdge;
    const auto sampler = device.createSamplerUnique(samplerInfo);
    PipelineState definition;
    definition.pipelineStateName = "Texture Debug Output";
    definition.vertexShaderName = "passthrough.vert";
    definition.pixelShaderName = "texture.frag";
    definition.a2vType = A2V_TYPES_SIMPLE;
    definition.depthTestMode = DEPTH_TEST_MODE::NONE;
    definition.bDepthWrite = false;
    definition.flag = PIPELINE_FLAG::DEBUG_DRAW;
    for (const auto& [pipeline, shader] : {std::pair{"BASIC", "basic"}, std::pair{"PBR", "pbr"}}) {
        auto scene = definition;
        scene.pipelineStateName = pipeline;
        scene.vertexShaderName = std::string(shader) + ".vert";
        scene.pixelShaderName = std::string(shader) + ".frag";
        scene.a2vType = A2V_TYPES_FULL;
        scene.flag = PIPELINE_FLAG::NONE;
        const auto state = Register(manager, scene);
        manager.CreateGraphicsPipeline(*state, *renderPass);
    }
    for (const bool frontCull : {false, true}) {
        definition.pipelineStateName = frontCull ? "Front-cull regression" : "Texture Debug Output";
        definition.cullFaceMode = frontCull ? CULL_FACE_MODE::FRONT : CULL_FACE_MODE::BACK;
        const auto state = Register(manager, definition);
        manager.CreateGraphicsPipeline(*state, *renderPass);
        Descriptors descriptors(device, *state);
        const vk::DescriptorImageInfo image({}, resources.source.descriptor,
                                            vk::ImageLayout::eShaderReadOnlyOptimal);
        const vk::DescriptorImageInfo samplerImage(*sampler, {}, {});
        const std::array writes{
            vk::WriteDescriptorSet(descriptors.set, VulkanShaderBindings::kTextureBase,
                0, 1, vk::DescriptorType::eSampledImage, &image),
            vk::WriteDescriptorSet(descriptors.set, VulkanShaderBindings::kSamplerBase,
                0, 1, vk::DescriptorType::eSampler, &samplerImage)};
        device.updateDescriptorSets(writes, {});
        rhi.ExecuteCommands([&](vk::CommandBuffer command) {
            rhi.TransitionTexture(command, resources.target,
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}, vk::ImageLayout::eColorAttachmentOptimal);
            const vk::ClearValue clear(vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1}));
            command.beginRenderPass({*renderPass, *framebuffer, {{0, 0}, {16, 16}}, clear},
                                     vk::SubpassContents::eInline);
            command.setViewport(0, vk::Viewport(0, 16, 16, -16, 0, 1));
            command.setScissor(0, vk::Rect2D({0, 0}, {16, 16}));
            command.bindPipeline(vk::PipelineBindPoint::eGraphics, *state->pipelineState);
            const std::array<uint32_t, 4> offsets{};
            command.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *state->pipelineLayout,
                                       0, descriptors.set, offsets);
            const std::array buffers{resources.vertices[0].buffer, resources.vertices[1].buffer};
            const std::array<vk::DeviceSize, 2> vertexOffsets{};
            command.bindVertexBuffers(0, buffers, vertexOffsets);
            command.draw(6, 1, 0, 0);
            command.endRenderPass();
        });
        const auto pixels = rhi.ReadTexture(resources.target);
        Require(pixels.size() == 16 * 16 * 4, "Offscreen readback size mismatch");
        for (size_t y = 0; y < 16; ++y) {
            for (size_t x = 0; x < 16; ++x) {
                for (size_t channel = 0; channel < 4; ++channel) {
                    const auto texel = (y < 8 ? 0 : 2) + (x < 8 ? 0 : 1);
                    const auto expected = frontCull ? (channel == 3 ? 255 : 0)
                                                    : colors[texel * 4 + channel];
                    if (pixels[(y * 16 + x) * 4 + channel] != expected) {
                        const auto pixel = (y * 16 + x) * 4;
                        throw std::runtime_error("Offscreen mismatch at " + std::to_string(x) + "," +
                            std::to_string(y) + ", frontCull=" + std::to_string(frontCull) +
                            ", rgba=" + std::to_string(pixels[pixel]) + "," +
                            std::to_string(pixels[pixel + 1]) + "," + std::to_string(pixels[pixel + 2]) +
                            "," + std::to_string(pixels[pixel + 3]));
                    }
                }
            }
        }
        manager.UnregisterPipelineState(definition);
        Require(!state->pipelineState && !state->pipelineLayout && state->stages.empty(),
                "Unregistering a source definition retained GPU resources in the borrowed state");
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "Expected the generated asset root");
        VulkanRHI rhi;
        rhi.InitializeHeadless();
        {
            AssetLoader assets;
            assets.AddSearchPath(argv[1]);
            VulkanPipelineStateManager manager(rhi, assets);
            CheckFailures(rhi, manager);
            const auto retained = CheckCompute(rhi, manager);
            CheckOffscreen(rhi, manager);
            manager.Finalize();
            manager.Finalize();
            Require(!retained->pipelineState && !retained->pipelineLayout &&
                        retained->stages.empty() && !retained->descriptorSetLayout,
                    "Borrowed pipeline state retained GPU resources after manager finalization");
        }
        rhi.destroyAll();
        Require(rhi.GetValidationErrorCount() == 0, "Production pipeline regression reported validation errors");
        std::cout << "Production shared-shader loading, BRDF dispatch, offscreen orientation/culling, "
                     "failure cleanup, and retained-state teardown passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan production pipeline regression failed: " << error.what() << '\n';
        return 1;
    }
}
