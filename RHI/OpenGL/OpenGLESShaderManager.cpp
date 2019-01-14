#include "OpenGLESShaderManager.hpp"

#if defined(OS_WEBASSEMBLY)
#define SHADER_ROOT "Shaders/WebGL/"
#else
#define SHADER_ROOT "Shaders/OpenGLES/"
#endif

#include "OpenGLShaderManagerCommonBase.cpp"