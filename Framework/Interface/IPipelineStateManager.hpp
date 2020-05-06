#pragma once
#include "IRuntimeModule.hpp"
#include "portable.hpp"
#include "cbuffer.h"
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
        LESS,
        NOT_EQUAL,
        NEVER,
        ALWAYS
    };

    ENUM(STENCIL_TEST_MODE)
    {
        NONE
    };

    ENUM(CULL_FACE_MODE)
    {
        NONE,
        FRONT,
        BACK
    };

    ENUM(PIPELINE_TYPE)
    {
        GRAPHIC,
        COMPUTE
    };

    struct PipelineState
    {
        std::string pipelineStateName;
        PIPELINE_TYPE pipelineType;

        std::string vertexShaderName;
        std::string pixelShaderName;
        std::string computeShaderName;
        std::string geometryShaderName;
        std::string tessControlShaderName;
        std::string tessEvaluateShaderName;
        std::string meshShaderName;

        DEPTH_TEST_MODE depthTestMode;
        bool    bDepthWrite;
        STENCIL_TEST_MODE stencilTestMode;
        CULL_FACE_MODE  cullFaceMode;

        A2V_TYPES a2vType;

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