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

    VulkanPipelineState(PipelineState& state) : PipelineState(state) {}
};

class VulkanPipelineStateManager : public PipelineStateManager {
   public:
    VulkanPipelineStateManager() = default;
    ~VulkanPipelineStateManager() = default;

   protected:
    bool InitializePipelineState(PipelineState** ppPipelineState) final;
    void DestroyPipelineState(PipelineState& pipelineState) final;
};
}  // namespace My