#pragma once
#include "GameLogic.hpp"

namespace My {
    class BilliardGameLogic : public GameLogic
    {
        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();
    };
}