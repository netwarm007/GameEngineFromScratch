#if defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#elif defined(OS_WINDOWS)
#include "RHI/D3d/D3d12Config.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif
