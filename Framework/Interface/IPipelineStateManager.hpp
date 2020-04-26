#pragma once
#include "IRuntimeModule.hpp"
#include "portable.hpp"
#include <memory>
#include <string>

namespace My {
    ENUM(DEPTH_TEST_MODE)
    {
        NONE,
        LARGE,
        LARGE_EQUAL,
        EQUAL,
        LESS_EQUAL,
        LESS
    };

    ENUM(STENCIL_TEST_MODE)
    {
        NONE
    };

    struct PipelineState
    {
        std::string pipelineStateName;
        std::string vertexShaderName;
        std::string pixelShaderName;
        std::string computeShaderName;
        std::string geometryShaderName;
        std::string tessControlShaderName;
        std::string tessEvaluateShaderName;
        std::string meshShaderName;

        DEPTH_TEST_MODE depthTestMode;
        STENCIL_TEST_MODE stencilTestMode;

        virtual ~PipelineState() = default;
    }; 

    Interface IPipelineStateManager : inheritance IRuntimeModule
    {
    public:
        virtual bool RegisterPipelineState(PipelineState& pipelineState) = 0;
        virtual void UnregisterPipelineState(PipelineState& pipelineState) = 0;
        virtual void Clear() = 0;

        [[nodiscard]] virtual const std::shared_ptr<PipelineState> GetPipelineState(std::string name) const = 0;
    };

    extern IPipelineStateManager* g_pPipelineStateManager;
}