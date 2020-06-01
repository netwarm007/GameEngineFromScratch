#pragma once
#include "FrameStructure.hpp"
#include "IPass.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IDrawPass : _inherits_ IPass {
   public:
    IDrawPass() = default;
    virtual ~IDrawPass() = default;

    virtual void BeginPass(Frame & frame) = 0;
    virtual void Draw(Frame & frame) = 0;
    virtual void EndPass(Frame & frame) = 0;
};
}  // namespace My
