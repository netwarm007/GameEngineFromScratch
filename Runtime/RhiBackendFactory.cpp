#include "RhiComposition.hpp"

#include "config.h"

#if defined(OS_WINDOWS)
#include "Platform/Windows/D3d12Backend.hpp"
#include "Platform/Windows/VulkanApplication.hpp"
#include "Platform/Windows/Win32OpenGLBackend.hpp"
#include "Platform/Windows/WindowsApplication.hpp"
#elif defined(OS_MACOS)
#include "Platform/Darwin/CocoaApplication.h"
#include "Platform/Darwin/MetalBackend.h"
#include "Platform/Darwin/VulkanApplication.hpp"
#elif defined(OS_LINUX)
#include "Platform/Linux/VulkanApplication.hpp"
#include "Platform/Linux/XcbApplication.hpp"
#if defined(HAS_SDL2)
#include "Platform/Linux/LinuxHost.hpp"
#include "Platform/Sdl/SDLOpenGLBackend.hpp"
#endif
#endif

namespace My {

RhiBackendResult CreateRhiBackendObject(RhiBackend backend,
                                        BaseApplication& host) {
    RhiBackendResult result;
    switch (backend) {
        case RhiBackend::D3d12:
#if defined(OS_WINDOWS)
            result.backend = std::make_unique<D3d12Backend>(
                static_cast<WindowsApplication&>(host),
                static_cast<WindowsApplication&>(host));
#else
            result.error = "D3D12 is a Windows backend";
#endif
            break;
        case RhiBackend::OpenGL:
            result.fixOpenGLPerspectiveMatrix = true;
#if defined(OS_WINDOWS)
            result.backend = std::make_unique<Win32OpenGLBackend>(
                static_cast<WindowsApplication&>(host),
                static_cast<WindowsApplication&>(host));
#elif defined(OS_LINUX) && defined(HAS_SDL2)
            {
                auto& linuxHost = static_cast<LinuxHost&>(host);
                result.backend = std::make_unique<SDLOpenGLBackend>(
                    linuxHost, linuxHost,
                    [&linuxHost] { return linuxHost.GetSdlWindow(); },
                    [&linuxHost](Uint32 flags) { linuxHost.SetWindowFlags(flags); });
            }
#else
            result.error =
                "OpenGL switching requires the SDL host in this build";
#endif
            break;
        case RhiBackend::Metal:
#if defined(OS_MACOS)
            result.backend = std::make_unique<MetalBackend>(
                host, static_cast<CocoaApplication&>(host),
                [&host] { return host.GetInputManager(); });
#else
            result.error = "Metal is a macOS backend";
#endif
            break;
        case RhiBackend::Vulkan:
#if defined(OS_WINDOWS)
            result.backend = MakeWin32VulkanBackend(
                static_cast<WindowsApplication&>(host));
#elif defined(OS_MACOS)
            result.backend = MakeCocoaVulkanBackend(
                static_cast<CocoaApplication&>(host));
#elif defined(OS_LINUX) && defined(HAS_SDL2)
            {
                auto& linuxHost = static_cast<LinuxHost&>(host);
                result.backend = MakeXcbVulkanBackend(
                    linuxHost, linuxHost, linuxHost.GetXcbConnection());
            }
#elif defined(OS_LINUX)
            {
                auto& xcbHost = static_cast<XcbApplication&>(host);
                result.backend = MakeXcbVulkanBackend(
                    xcbHost, xcbHost, xcbHost.GetXcbConnection());
            }
#else
            result.error = "Vulkan support was not configured";
#endif
            break;
    }
    if (result.backend) {
        result.recreateWindowOnReconstruction =
            result.backend->RequiresWindowRecreation();
    }
    return result;
}

}  // namespace My