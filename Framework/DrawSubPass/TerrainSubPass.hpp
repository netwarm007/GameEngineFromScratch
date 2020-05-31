#pragma once
#include "BaseSubPass.hpp"

namespace My {
class TerrainSubPass : public BaseSubPass {
   public:
    void Draw(Frame& frame) final;
};
}  // namespace My