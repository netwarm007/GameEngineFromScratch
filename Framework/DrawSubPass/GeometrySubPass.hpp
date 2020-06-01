#pragma once
#include "BaseSubPass.hpp"

namespace My {
class GeometrySubPass : public BaseSubPass {
   public:
    void Draw(Frame& frame) final;
};
}  // namespace My
