#pragma once
#include "IDrawPass.hpp"

namespace My {
    class TerrainPass : implements IDrawPass
    {
    public:
        ~TerrainPass() = default;
        void Draw(Frame& frame) final;
    };
}