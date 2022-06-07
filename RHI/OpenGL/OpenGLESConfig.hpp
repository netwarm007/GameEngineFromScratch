#pragma once
#include "OpenGL/OpenGLESGraphicsManager.hpp"
#include "OpenGL/OpenGLESPipelineStateManager.hpp"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new OpenGLESGraphicsManager);
PipelineStateManager* g_pShaderManager =
    static_cast<PipelineStateManager*>(new OpenGLESPipelineStateManager);
}  // namespace My
