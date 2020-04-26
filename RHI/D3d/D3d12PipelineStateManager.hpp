#pragma once
#include <d3d12.h>
#include "PipelineStateManager.hpp"
#include "cbuffer.h"

namespace My {
    struct D3d12PipelineState : public PipelineState
    {
        D3D12_SHADER_BYTECODE vertexShaderByteCode;
        D3D12_SHADER_BYTECODE pixelShaderByteCode;
        D3D12_SHADER_BYTECODE geometryShaderByteCode;
        D3D12_SHADER_BYTECODE computeShaderByteCode;
        int32_t psoIndex;
        A2V_TYPES a2vType;

        D3d12PipelineState()
        {
            psoIndex = -1;
        }
    };

    class D3d12PipelineStateManager : public PipelineStateManager
    {
    public:
        D3d12PipelineStateManager() = default;
        ~D3d12PipelineStateManager() = default;

    protected:
        virtual bool InitializePipelineState(PipelineState** ppPipelineState) final;
        virtual void DestroyPipelineState(PipelineState& pipelineState) final;
    };
}