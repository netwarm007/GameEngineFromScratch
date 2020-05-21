#pragma once
#include "IDrawPhase.hpp"

namespace My {
class SkyBoxPhase : _implements_ IDrawPhase {
   public:
    ~SkyBoxPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My