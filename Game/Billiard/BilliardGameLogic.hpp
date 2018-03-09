#pragma once
#include "IGameLogic.hpp"

namespace My {
    class BilliardGameLogic : implements IGameLogic
    {
        int Initialize();
        void Finalize();
        void Tick();

        void OnLeftKey();
    };
}