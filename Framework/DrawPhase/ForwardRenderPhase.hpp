#pragma once
#include "IDrawPhase.hpp"

namespace My {
    class ForwardRenderPhase : implements IDrawPhase
    {
    public:
        ~ForwardRenderPhase() = default;
        void Draw(Frame& frame) final;
    };
}
