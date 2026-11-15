#pragma once

#include <span>

#include "IRuntimeModule.hpp"

namespace My {
// The caller supplies live, initialized modules in the desired frame order.
inline void TickFrame(std::span<IRuntimeModule* const> modules) {
    for (auto* module : modules) {
        module->Tick();
    }
}
}  // namespace My
