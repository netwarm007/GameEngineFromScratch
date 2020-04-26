#include "OpenGLESPipelineStateManager.hpp"

#if defined(OS_WEBASSEMBLY)
#define SHADER_ROOT "Shaders/WebGL/"
#else
#define SHADER_ROOT "Shaders/OpenGLES/"
#endif

#if defined(OS_WEBASSEMBLY)
// disable compute shader
#define GLAD_GL_ARB_compute_shader 0
#endif

#include "OpenGLPipelineStateManagerCommonBase.cpp"