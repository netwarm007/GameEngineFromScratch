#include "GfxConfiguration.hpp"
#include "config.h"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#elif defined(OS_WINDOWS)
#include "D3d12Application.hpp"
#include "OpenGLApplication.hpp"
#else
#include "OpenGLApplication.hpp"
#endif

#if defined(HAS_VULKAN)
#include "VulkanApplication.hpp"
#endif

using namespace My;

int test(BaseApplication& app) {
    auto result = app.Initialize();

    if (result == 0) {
        app.CreateMainWindow();
        while (!app.IsQuit()) {
            app.Tick();
        }

        app.Finalize();
    }

    return result;
}

int main() {
    int result;

#if defined(OS_MACOS)
    {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "Cocoa Metal Application Test");
        CocoaMetalApplication app(config);
        result |= test(app);
    }
#endif

#if defined(OS_WINDOWS)
    {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "DX12 Application Test");
        D3d12Application app(config);
        result |= test(app);
    }
#endif

#if !defined(OS_MACOS)
    {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "OpenGL Application Test");
        OpenGLApplication app(config);
        result |= test(app);
    }
#endif

#if defined(HAS_VULKAN)
    {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "Vulkan Application Test");
        VulkanApplication app(config);
        result |= test(app);
    }
#endif

    return result;
}