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
#else
#include "OpenGLApplication.hpp"
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