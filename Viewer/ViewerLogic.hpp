#pragma once
#include "IGameLogic.hpp"

namespace My {
class ViewerLogic : _implements_ IGameLogic {
    // overrides
    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void OnLeftKeyDown() final;
    void OnRightKeyDown() final;
    void OnUpKeyDown() final;
    void OnDownKeyDown() final;

    void OnAnalogStick(int id, float deltaX, float deltaY) final;
#ifdef DEBUG
    void DrawDebugInfo() final;
#endif
};
}  // namespace My
