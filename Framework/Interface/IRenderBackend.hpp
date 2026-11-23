#pragma once

#include "Interface.hpp"

namespace My {
class IApplication;
class IPlatformHost;

// Backend-native lifecycle implemented by the session-owned backend object
// against a persistent platform host (M4.1 detachment). The host keeps
// window ownership and event routing; the backend owns device/swapchain/
// drawable state and the backend-specific GUI platform binding. The
// persistent host calls these in dependency order: graphics resources
// before GUI platform binding and module initialization; on teardown,
// modules finalize first, then GUI platform unbind, then graphics
// resources, then platform resources.
_Interface_ IRenderBackend {
   public:
    virtual ~IRenderBackend() = default;

    // Create the native window. Backends that constrain creation (GLX
    // visual selection, SDL GL attributes) implement it against their host;
    // others delegate to the host's generic creation.
    virtual void CreateMainWindow() = 0;

    // Create backend-native device/swapchain/drawable state on the host
    // window. On failure, release partial initialization before returning.
    virtual int InitializeGraphicsResources() = 0;

    // Drain all render/compute/presentation work before teardown or failure
    // rollback releases module resources.
    virtual void WaitForGraphicsIdle() = 0;

    // Release backend-native state. Must be safe after partial
    // initialization and safe to call repeatedly.
    virtual void FinalizeGraphicsResources() = 0;

    // Bind/unbind the GUI platform backend. Backends with special
    // requirements (SDL InitForOpenGL, Win32 InitForOpenGL + WGL viewport
    // callbacks, Darwin Vulkan surface callback) implement them; others
    // delegate to the host's generic binding.
    virtual int InitializeGuiPlatform() = 0;
    virtual void FinalizeGuiPlatform() = 0;

    // Present the completed frame to the native window (buffer swap) after
    // the module frame. Backends that present inside the graphics manager
    // keep the default no-op.
    virtual void PresentFrame() {}

    // Native window resize notification from the host (swapchain/drawable
    // recreation where the backend requires it).
    virtual void OnWindowResize(int new_width, int new_height) {}

    // Called after the session modules activated successfully: the renderer
    // ImGui backend is bound here, so renderer-dependent platform setup
    // (e.g. WGL multi-viewport callbacks) belongs in this hook.
    virtual void OnSessionActivated() {}

    // Whether reconstruction/switching must recreate the native window
    // (WGL pixel format, GLX visual, SDL context are window-lifetime bound).
    virtual bool RequiresWindowRecreation() const { return false; }
};
}  // namespace My