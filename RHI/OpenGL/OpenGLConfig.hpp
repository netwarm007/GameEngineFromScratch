#pragma once
#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "OpenGL/OpenGLPipelineStateManager.hpp"
#define IS_OPENGL 1

namespace My {
using TGraphicsManager = OpenGLGraphicsManager;
using TPipelineStateManager = OpenGLPipelineStateManager;
}  // namespace My
