#pragma once
#include "FrameStructure.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IPass {
   public:
    virtual void BeginPass() = 0;
    virtual void EndPass() = 0;
};
}  // namespace My