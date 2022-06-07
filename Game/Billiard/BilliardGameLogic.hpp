#pragma once
#include "GameLogic.hpp"

namespace My {
class BilliardGameLogic : public GameLogic {
    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void OnLeftKey() final;
};
}  // namespace My