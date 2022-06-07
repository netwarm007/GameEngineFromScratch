#pragma once
#ifdef DEBUG
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IDebugManager : _inherits_ IRuntimeModule {
   public:
    IDebugManager() = default;
    virtual ~IDebugManager() = default;

    virtual void ToggleDebugInfo() = 0;
};
}  // namespace My

#endif