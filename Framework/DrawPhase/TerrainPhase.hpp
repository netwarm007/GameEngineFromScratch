#pragma once
#include "IDrawPhase.hpp"

namespace My {
    class TerrainPhase : implements IDrawPhase
    {
    public:
        ~TerrainPhase() = default;
        void Draw(Frame& frame) final;
    };
}