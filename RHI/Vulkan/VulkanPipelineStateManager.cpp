#include "VulkanPipelineStateManager.hpp"

bool VulkanPipelineStateManager::InitializePipelineState(
    PipelineState** ppPipelineState) {
    VulkanPipelineState* pState = new VulkanPipelineState(**ppPipelineState);

    //loadShaders(pState);

    *ppPipelineState = pState;

    return true;
}

void VulkanPipelineStateManager::DestroyPipelineState(
    PipelineState& pipelineState) {
    VulkanPipelineState* pPipelineState =
        dynamic_cast<VulkanPipelineState*>(&pipelineState);

    delete pPipelineState;
}
