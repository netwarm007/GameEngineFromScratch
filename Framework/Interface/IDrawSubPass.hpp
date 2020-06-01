#pragma once
#include "FrameStructure.hpp"
#include "ISubPass.hpp"

namespace My {
_Interface_ IDrawSubPass : _inherits_ ISubPass {
   public:
    virtual void Draw(Frame & frame) = 0;
};
}  // namespace My
