#pragma once
#include "IDrawPhase.hpp"

namespace My {
class ForwardRenderPhase : implements IDrawPhase {
   public:
    ~ForwardRenderPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My
