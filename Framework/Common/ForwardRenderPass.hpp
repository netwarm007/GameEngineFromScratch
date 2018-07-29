#pragma once
#include "IDrawPass.hpp"

namespace My {
    class ForwardRenderPass : implements IDrawPass
    {
    public:
        ~ForwardRenderPass() = default;
        void Draw(Frame& frame) final;
    };
}
