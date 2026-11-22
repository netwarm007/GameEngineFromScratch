#include "GfxConfiguration.hpp"
#include "config.h"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#elif defined(OS_WINDOWS)
#include "D2dApplication.hpp"
#include "D3d12Application.hpp"
#include "OpenGLApplication.hpp"
#elif defined(OS_ANDROID)
#if defined(HAS_VULKAN)
#include "VulkanApplication.hpp"
#else
#include "OpenGLESApplication.hpp"
#endif
#elif defined(HAS_SDL2)
// Pair with the runtime composition's SDL host; the GLX header pulls in
// X11, whose None macro collides with engine enums in mixed translation
// units.
#include "Platform/Sdl/OpenGLApplication.hpp"
#else
#include "Platform/Linux/OpenGLApplication.hpp"
#endif

namespace My {
static BaseApplication* CreateApplication(GfxConfiguration config) {
#if defined(OS_MACOS)
    return new CocoaMetalApplication(config);
#endif

#if defined(OS_WINDOWS)
    return new D3d12Application(config);
#endif

#if defined(OS_ANDROID)
#if defined(HAS_VULKAN)
    return new VulkanApplication(config);
#else
    return new OpenGLESApplication(config);
#endif
#endif

#if defined(OS_LINUX)
    return new OpenGLApplication(config);
#endif
}
}  // namespace My