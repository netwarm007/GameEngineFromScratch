#pragma once
#include "IDrawPhase.hpp"

namespace My {
class SkyBoxPhase : implements IDrawPhase {
   public:
    ~SkyBoxPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My