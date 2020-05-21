#pragma once
#include <d3d12.h>

#include "PipelineStateManager.hpp"

namespace My {
struct D3d12PipelineState : public PipelineState {
    D3D12_SHADER_BYTECODE vertexShaderByteCode;
    D3D12_SHADER_BYTECODE pixelShaderByteCode;
    D3D12_SHADER_BYTECODE geometryShaderByteCode;
    D3D12_SHADER_BYTECODE computeShaderByteCode;
    int32_t psoIndex{-1};

    D3d12PipelineState(PipelineState& state) : PipelineState(state) {}
};

class D3d12PipelineStateManager : public PipelineStateManager {
   public:
    D3d12PipelineStateManager() = default;
    ~D3d12PipelineStateManager() = default;

   protected:
    bool InitializePipelineState(PipelineState** ppPipelineState) final;
    void DestroyPipelineState(PipelineState& pipelineState) final;
};
}  // namespace My