#pragma once
#include "IDrawPhase.hpp"

namespace My {
    class HUDPhase : implements IDrawPhase
    {
    public:
        ~HUDPhase() = default;
        void Draw(Frame& frame) final;
    };
}
