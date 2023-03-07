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
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600, "Basic Window Test");
#if defined(OS_MACOS)
BaseApplication* g_pApp = new CocoaApplication(config);
#elif defined(OS_WINDOWS)
BaseApplication* g_pApp = new WindowsApplication(config);
#elif defined(OS_ANDROID)
BaseApplication* g_pApp = new AndroidApplication(config);
#else
BaseApplication* g_pApp = new XcbApplication(config);
#endif
} // namespace My

using namespace My;

int main() {
    int result;

    result = g_pApp->Initialize();

    g_pApp->CreateMainWindow();

    while(!g_pApp->IsQuit()) {
        g_pApp->Tick();
    }

    g_pApp->Finalize();

    return result;
}