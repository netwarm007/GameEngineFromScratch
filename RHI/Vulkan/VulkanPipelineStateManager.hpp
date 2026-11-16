#pragma once
#include <vulkan/vulkan.hpp>

#include "PipelineStateManager.hpp"

namespace My {
struct VulkanPipelineState : public PipelineState {
    vk::ShaderModule vertexShaderByteCode;
    vk::ShaderModule pixelShaderByteCode;
    vk::ShaderModule geometryShaderByteCode;
    vk::ShaderModule computeShaderByteCode;
    vk::ShaderModule meshShaderByteCode;
    vk::Pipeline pipelineState{nullptr};
    vk::PipelineLayout pipelineLayout{nullptr};
    vk::DescriptorSetLayout descriptorSetLayout{nullptr};
    bool supportsSceneGeometry{false};
    bool supportsSkyBox{false};
    bool supportsShadowMap{false};
    bool supportsCompute{false};
    bool shadowMapUsesFragmentShader{false};

    VulkanPipelineState(PipelineState& state) : PipelineState(state) {}
};

class VulkanPipelineStateManager : public PipelineStateManager {
   public:
    VulkanPipelineStateManager() = default;
    ~VulkanPipelineStateManager() = default;

    int Initialize() override;
    void UnregisterPipelineState(PipelineState& pipelineState) override;
    void InvalidatePipelines();
    [[nodiscard]] std::shared_ptr<PipelineState>
    GetSceneGeometryPipelineState() const;

   protected:
    bool InitializePipelineState(PipelineState** ppPipelineState) final;
    void DestroyPipelineState(PipelineState& pipelineState) final;
};
}  // namespace My