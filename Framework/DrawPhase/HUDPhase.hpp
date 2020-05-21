#pragma once
#include "IDrawPhase.hpp"

namespace My {
class HUDPhase : _implements_ IDrawPhase {
   public:
    ~HUDPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My
