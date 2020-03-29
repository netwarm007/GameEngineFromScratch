#pragma once
#include "IDrawPhase.hpp"

namespace My {
    class HUDPhase : implements IDrawPhase
    {
    public:
        ~HUDPhase() override = default;
        void Draw(Frame& frame) final;
    };
}
