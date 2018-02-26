#pragma once
#include "GameLogic.hpp"

namespace My {
    class article_44_logic : public GameLogic
    {
        int Initialize();
        void Finalize();
        void Tick();

        void OnLeftKey();
        void OnRightKey();
        void OnUpKey();
        void OnDownKey();
    };
}