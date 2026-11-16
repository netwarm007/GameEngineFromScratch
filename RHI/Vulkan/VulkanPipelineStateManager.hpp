#pragma once
#include "IAssetReader.hpp"
#include "PipelineStateManager.hpp"
#include "VulkanRHI.hpp"

namespace My {
struct VulkanPipelineState : public PipelineState {
    struct Stage {
        vk::UniqueShaderModule module;
        vk::ShaderStageFlagBits stage;
        std::string entry;
    };

    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    std::vector<Stage> stages;
    vk::UniquePipeline pipelineState;

    explicit VulkanPipelineState(const PipelineState& state) : PipelineState(state) {}
};

class VulkanPipelineStateManager : public PipelineStateManager {
   public:
    VulkanPipelineStateManager(VulkanRHI& rhi, IAssetReader& assets)
        : m_rhi(rhi), m_assets(assets) {}

    int Initialize() final;
    void CreateGraphicsPipeline(VulkanPipelineState& state, vk::RenderPass renderPass);
    void UnregisterPipelineState(PipelineState& pipelineState) final;

   protected:
    bool InitializePipelineState(PipelineState** ppPipelineState) final;
    void DestroyPipelineState(PipelineState& pipelineState) final;

   private:
    VulkanRHI& m_rhi;
    IAssetReader& m_assets;
};
}  // namespace My