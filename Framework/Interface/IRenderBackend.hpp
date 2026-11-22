#pragma once

#include "Interface.hpp"

namespace My {
// Backend-native lifecycle hooks implemented by the render session's backend
// side. The persistent host calls these in dependency order: graphics
// resources before GUI platform binding and module initialization; on
// teardown, modules finalize first, then GUI platform unbind, then graphics
// resources, then platform resources.
_Interface_ IRenderBackend {
   public:
    virtual ~IRenderBackend() = default;

    // Create backend-native device/swapchain/drawable state on the host
    // window. On failure, release partial initialization before returning.
    virtual int InitializeGraphicsResources() = 0;

    // Drain all render/compute/presentation work before teardown or failure
    // rollback releases module resources.
    virtual void WaitForGraphicsIdle() = 0;

    // Release backend-native state. Must be safe after partial
    // initialization and safe to call repeatedly.
    virtual void FinalizeGraphicsResources() = 0;
};
}  // namespace My