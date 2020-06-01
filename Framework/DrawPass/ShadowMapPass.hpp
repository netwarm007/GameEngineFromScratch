#pragma once
#include "BasePass.hpp"

namespace My {
class ShadowMapPass : public BasePass {
   public:
    ~ShadowMapPass() override = default;

    void BeginPass(Frame& frame) override {}
    void Draw(Frame& frame) final;
    void EndPass(Frame& frame) override {}
};
}  // namespace My
