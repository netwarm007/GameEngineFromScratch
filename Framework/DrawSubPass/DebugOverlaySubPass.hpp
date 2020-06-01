#pragma once
#include "BaseSubPass.hpp"

namespace My {
class DebugOverlaySubPass : public BaseSubPass {
   public:
    void Draw(Frame& frame) final;
};
}  // namespace My
