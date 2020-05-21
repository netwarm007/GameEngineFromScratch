#pragma once
#include "PipelineStateManager.hpp"

namespace My {
struct OpenGLPipelineState : public PipelineState {
    uint32_t shaderProgram = 0;
    OpenGLPipelineState(PipelineState& rhs) : PipelineState(rhs) {}
    OpenGLPipelineState(PipelineState&& rhs) : PipelineState(std::move(rhs)) {}
};

class OpenGLPipelineStateManagerCommonBase : public PipelineStateManager {
   public:
    OpenGLPipelineStateManagerCommonBase() = default;
    virtual ~OpenGLPipelineStateManagerCommonBase() = default;

   protected:
    bool InitializePipelineState(PipelineState** ppPipelineState) final;
    void DestroyPipelineState(PipelineState& pipelineState) final;
};
}  // namespace My