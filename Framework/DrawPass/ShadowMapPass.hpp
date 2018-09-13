#pragma once
#include "IDrawPass.hpp"

namespace My {
    class ShadowMapPass: implements IDrawPass
    {
    public:
        ~ShadowMapPass() = default;
        void Draw(Frame& frame) final;
    };
}
