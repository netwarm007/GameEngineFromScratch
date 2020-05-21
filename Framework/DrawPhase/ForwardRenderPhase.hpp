#pragma once
#include "IDrawPhase.hpp"

namespace My {
class ForwardRenderPhase : _implements_ IDrawPhase {
   public:
    ~ForwardRenderPhase() override = default;
    void Draw(Frame& frame) final;
};
}  // namespace My
