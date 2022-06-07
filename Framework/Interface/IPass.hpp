#pragma once
#include "FrameStructure.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IPass {
   public:
    virtual ~IPass() = default;
    virtual void BeginPass(Frame & frame) = 0;
    virtual void EndPass(Frame & frame) = 0;
};
}  // namespace My