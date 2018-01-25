#pragma once
#include "IRuntimeModule.hpp"

namespace My {
    class GameLogic : implements IRuntimeModule
    {
        public:
        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();
    };

    extern GameLogic* g_pGameLogic;
}