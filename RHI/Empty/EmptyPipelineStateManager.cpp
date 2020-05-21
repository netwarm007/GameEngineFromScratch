#include "EmptyPipelineStateManager.hpp"

using namespace My;

bool EmptyPipelineStateManager::InitializePipelineState(
    PipelineState** ppPipelineState) {
    return true;
}

void EmptyPipelineStateManager::DestroyPipelineState(
    PipelineState& pipelineState) {}