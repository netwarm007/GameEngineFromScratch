#pragma once
#include "FrameStructure.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IDispatchPass {
   public:
    IDispatchPass() = default;
    virtual ~IDispatchPass() = default;

    virtual void Dispatch(Frame & frame) = 0;
};
}  // namespace My
