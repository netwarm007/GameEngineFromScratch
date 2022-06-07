#pragma once
#include "BaseSubPass.hpp"

namespace My {
class SkyBoxSubPass : public BaseSubPass {
   public:
    using BaseSubPass::BaseSubPass;
    void Draw(Frame& frame) final;
};
}  // namespace My