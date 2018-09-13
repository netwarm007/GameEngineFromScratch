#pragma once
#include "IDrawPass.hpp"

namespace My {
    class SkyBoxPass : implements IDrawPass
    {
    public:
        ~SkyBoxPass() = default;
        void Draw(Frame& frame) final;
    };
}