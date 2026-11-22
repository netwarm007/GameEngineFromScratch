#pragma once

#include <cstdint>

#include "Interface.hpp"

namespace My {
// Persistent platform host surface consumed by a render session: GUI
// platform backend binding and platform event processing, separate from
// module frame updates. Window ownership and the window/configuration
// surface remain on IApplication, which the host also implements; the host
// object persists across render-session replacement.
_Interface_ IPlatformHost {
   public:
    virtual ~IPlatformHost() = default;

    // Bind/unbind the GUI platform backend to the native window. The ImGui
    // context must exist and be unbound before InitializeGuiPlatform.
    virtual int InitializeGuiPlatform() = 0;
    virtual void FinalizeGuiPlatform() = 0;

    // Drain pending platform events and run GUI platform per-frame work
    // without ticking runtime modules. Safe to call while rendering is
    // paused.
    virtual void ProcessEvents() = 0;
};
}  // namespace My