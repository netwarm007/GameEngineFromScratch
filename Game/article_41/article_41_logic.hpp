#pragma once
#include "GameLogic.hpp"

namespace My {
    class article_41_logic : public GameLogic
    {
        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

        virtual void OnLeftKey();
    };
}