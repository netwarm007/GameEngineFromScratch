#pragma once
#include "IDrawPhase.hpp"

namespace My {
class TerrainPhase : implements IDrawPhase {
   public:
    ~TerrainPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My