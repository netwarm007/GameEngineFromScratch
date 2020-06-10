#pragma once
#include "BaseDrawPass.hpp"

namespace My {
class RayTracePass : public BaseDrawPass {
    void BeginPass() final;
    void Draw(Frame& frame) final;
    void EndPass() final;
};
}  // namespace My