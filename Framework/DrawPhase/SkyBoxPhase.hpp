#pragma once
#include "IDrawPhase.hpp"

namespace My {
    class SkyBoxPhase : implements IDrawPhase 
    {
    public:
        ~SkyBoxPhase() = default;
        void Draw(Frame& frame) final;
    };
}