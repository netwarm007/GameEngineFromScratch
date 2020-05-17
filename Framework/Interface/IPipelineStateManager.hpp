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

    ENUM(PIPELINE_FLAG)
    {
        NONE,
        LIGHT_INFO,
        SHADOW_MATRIX,
        DEBUG_PARAM
    };

    ENUM(PIXEL_FORMAT)
    {
        INVALID,
        BGRA8UNORM
    };

    struct PipelineState
    {
        std::string pipelineStateName;
        PIPELINE_TYPE pipelineType{PIPELINE_TYPE::GRAPHIC};

        std::string vertexShaderName;
        std::string pixelShaderName;
        std::string computeShaderName;
        std::string geometryShaderName;
        std::string tessControlShaderName;
        std::string tessEvaluateShaderName;
        std::string meshShaderName;

        DEPTH_TEST_MODE depthTestMode{DEPTH_TEST_MODE::ALWAYS};
        bool    bDepthWrite{true};
        STENCIL_TEST_MODE stencilTestMode{STENCIL_TEST_MODE::NONE};
        CULL_FACE_MODE  cullFaceMode{CULL_FACE_MODE::BACK};
        PIXEL_FORMAT pixelFormat{PIXEL_FORMAT::BGRA8UNORM};
        uint32_t    sampleCount{1};

        A2V_TYPES a2vType{A2V_TYPES::A2V_TYPES_NONE};
        PIPELINE_FLAG flag;

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