#include "config.h"
#include "GfxConfiguration.hpp"
#include "portable.hpp"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaApplication.h"
#elif defined(OS_WINDOWS)
#include "WindowsApplication.hpp"
#else
#include "XcbApplication.hpp"
#endif

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600, "Basic Window Test");
#if defined(OS_MACOS)
BaseApplication* g_pApp = new CocoaApplication(config);
#elif defined(OS_WINDOWS)
BaseApplication* g_pApp = new WindowsApplication(config);
#else
BaseApplication* g_pApp = new XcbApplication(config);
#endif
} // namespace My

using namespace My;

int main() {
    int result;

    g_pApp->CreateMainWindow();

    result = g_pApp->Initialize();

    while(!g_pApp->IsQuit()) {
        g_pApp->Tick();
    }

    g_pApp->Finalize();

    return result;
}