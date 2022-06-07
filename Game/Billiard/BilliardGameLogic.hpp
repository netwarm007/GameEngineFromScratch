#pragma once
#include "GameLogic.hpp"

namespace My {
class BilliardGameLogic : public GameLogic {
    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void OnLeftKey() final;
    void OnAnalogStick(int id, float deltaX, float deltaY) final;
};
}  // namespace My