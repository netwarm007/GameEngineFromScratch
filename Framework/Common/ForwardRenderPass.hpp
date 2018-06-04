#pragma once
#include "DrawPass.hpp"

namespace My {
    class ForwardRenderPass : implements DrawPass
    {
    public:
        ~ForwardRenderPass() = default;
        void Draw();
    };
}
