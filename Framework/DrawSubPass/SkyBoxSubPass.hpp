#pragma once
#include "BaseSubPass.hpp"

namespace My {
class SkyBoxSubPass : public BaseSubPass {
   public:
    void Draw(Frame& frame) final;
};
}  // namespace My