#pragma once
#include "FrameStructure.hpp"
#include "IPass.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IDispatchPass : _inherits_ IPass {
   public:
    virtual void Dispatch(Frame & frame) = 0;
};
}  // namespace My
