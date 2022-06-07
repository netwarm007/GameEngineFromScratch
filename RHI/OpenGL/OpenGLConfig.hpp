#pragma once
#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "OpenGL/OpenGLPipelineStateManager.hpp"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
PipelineStateManager* g_pPipelineStateManager =
    static_cast<PipelineStateManager*>(new OpenGLPipelineStateManager);
}  // namespace My
