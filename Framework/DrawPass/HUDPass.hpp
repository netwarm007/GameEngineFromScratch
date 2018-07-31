#pragma once
#include "IDrawPass.hpp"

namespace My {
    class HUDPass : implements IDrawPass
    {
    public:
        ~HUDPass() = default;
        void Draw(Frame& frame) final;
    };
}
