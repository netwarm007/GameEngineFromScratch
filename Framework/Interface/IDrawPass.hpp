#pragma once
#include "FrameStructure.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IDrawPass {
   public:
    IDrawPass() = default;
    virtual ~IDrawPass() = default;

    virtual void BeginPass(Frame & frame) = 0;
    virtual void Draw(Frame & frame) = 0;
    virtual void EndPass(Frame & frame) = 0;
};
}  // namespace My
