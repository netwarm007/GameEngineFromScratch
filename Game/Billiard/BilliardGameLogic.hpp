#pragma once
#include "IGameLogic.hpp"

namespace My {
class BilliardGameLogic : _implements_ IGameLogic {
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void OnLeftKey() override;
};
}  // namespace My