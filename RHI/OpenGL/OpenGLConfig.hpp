#pragma once
#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "OpenGL/OpenGLPipelineStateManager.hpp"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
IPipelineStateManager* g_pPipelineStateManager =
    static_cast<IPipelineStateManager*>(new OpenGLPipelineStateManager);
}  // namespace My
