#pragma once
#include "IGameLogic.hpp"

namespace My {
    class EditorLogic : implements IGameLogic
    {
        int Initialize();
        void Finalize();
        void Tick();
    };
}