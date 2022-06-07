#pragma once
#ifdef DEBUG
#include "IDebugManager.hpp"

namespace My {
class DebugManager : _implements_ IDebugManager {
   public:
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void ToggleDebugInfo() override;

    bool m_bDrawDebugInfo = false;
};
}  // namespace My

#endif