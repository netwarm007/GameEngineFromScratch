#pragma once
#ifdef DEBUG
#include "Interface.hpp"

namespace My {
_Interface_ IDebugManager {
   public:
    IDebugManager() = default;
    virtual ~IDebugManager() = default;

    virtual void ToggleDebugInfo() = 0;
};
}  // namespace My

#endif