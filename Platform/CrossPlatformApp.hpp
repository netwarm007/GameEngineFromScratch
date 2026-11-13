#include "config.h"
#include "GfxConfiguration.hpp"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/SdlApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaApplication.h"
#elif defined(OS_WINDOWS)
#include "WindowsApplication.hpp"
#elif defined(OS_ANDROID)
#include "AndroidApplication.hpp"
#else
#include "XcbApplication.hpp"
#endif

namespace My {
static BaseApplication* CreateApplication(GfxConfiguration config) {
#if defined(OS_MACOS)
    return new CocoaApplication(config);
#endif

#if defined(OS_WINDOWS)
    return new WindowsApplication(config);
#endif

#if defined(OS_ANDROID)
    return new AndroidApplication(config);
#endif

#if defined(OS_LINUX)
    return new XcbApplication(config);
#endif
}
}  // namespace My