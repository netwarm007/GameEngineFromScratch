#pragma once
#include "FrameStructure.hpp"
#include "IPass.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IDrawPass : _inherits_ IPass {
   public:
    virtual void Draw(Frame & frame) = 0;
};
}  // namespace My
