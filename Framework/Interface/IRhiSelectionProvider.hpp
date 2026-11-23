#pragma once
#include <string>
#include <vector>

#include "Interface.hpp"
#include "RhiStartup.hpp"

namespace My {
// M5.1: supplies the Viewer GUI with backend availability, the active backend,
// and a switch request entry point. Implemented by the application against the
// runtime RhiSwitchController; the GUI sub-pass queries it through the graphics
// manager. Kept free of Runtime-layer types so Framework can depend on it.
_Interface_ IRhiSelectionProvider {
   public:
    virtual ~IRhiSelectionProvider() = default;

    // Compiled inclusion + runtime availability for every matrix backend.
    virtual std::vector<RhiBackendAvailability> Backends() = 0;
    virtual RhiBackend ActiveBackend() = 0;

    // Queue a switch for the next frame boundary. Returns the
    // RhiSwitchRequestResult value as an int (Framework stays decoupled from
    // the Runtime enum); -1 when no controller is bound yet.
    virtual int RequestSwitch(RhiBackend target) = 0;

    // True while a switch is in progress (simulation paused / first-frame gate).
    virtual bool IsSwitching() = 0;
    // Human-readable status / last error for the progress and error UI.
    virtual std::string SwitchStatus() = 0;
};
}  // namespace My
