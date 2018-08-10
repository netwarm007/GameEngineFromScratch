#include "OpenGLESShaderManager.hpp"

#define VS_SHADER_SOURCE_FILE "Shaders/OpenGLES/basic_vs.glsl"
#define PS_SHADER_SOURCE_FILE "Shaders/OpenGLES/basic_ps.glsl"
#define VS_SHADOWMAP_SOURCE_FILE "Shaders/OpenGLES/shadowmap_vs.glsl"
#define PS_SHADOWMAP_SOURCE_FILE "Shaders/OpenGLES/shadowmap_ps.glsl"
#define DEBUG_VS_SHADER_SOURCE_FILE "Shaders/OpenGLES/debug_vs.glsl"
#define DEBUG_PS_SHADER_SOURCE_FILE "Shaders/OpenGLES/debug_ps.glsl"
#define VS_PASSTHROUGH_SOURCE_FILE "Shaders/OpenGLES/passthrough_vs.glsl"
#define PS_SIMPLE_TEXTURE_SOURCE_FILE "Shaders/OpenGLES/depthtexture_ps.glsl"
#define VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE "Shaders/OpenGLES/passthrough_cube_vs.glsl"
#define PS_SIMPLE_CUBEMAP_SOURCE_FILE "Shaders/OpenGLES/depthcubemap_ps.glsl"

#include "OpenGLShaderManagerCommonBase.cpp"