#pragma once
#include "BaseSubPass.hpp"

namespace My {
class GuiSubPass : public BaseSubPass {
   public:
    void Draw(Frame& frame) final;
};
}  // namespace My