#pragma once
#include "BaseDrawPass.hpp"

namespace My {
class ShadowMapPass : public BaseDrawPass {
   public:
    void BeginPass() override {}
    void Draw(Frame& frame) final;
    void EndPass() override {}
};
}  // namespace My
