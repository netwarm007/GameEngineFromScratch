#pragma once
#include "IDrawPhase.hpp"

namespace My {
class TerrainPhase : _implements_ IDrawPhase {
   public:
    ~TerrainPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My