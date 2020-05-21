#pragma once
#include "OpenGL/OpenGLESGraphicsManager.hpp"
#include "OpenGL/OpenGLESPipelineStateManager.hpp"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new OpenGLESGraphicsManager);
IPipelineStateManager* g_pShaderManager =
    static_cast<IPipelineStateManager*>(new OpenGLESPipelineStateManager);
}  // namespace My
