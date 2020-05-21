#pragma once
#include "FrameStructure.hpp"
#include "IPhase.hpp"

namespace My {
_Interface_ IDrawPhase : _inherits_ IPhase {
   public:
    IDrawPhase() = default;
    ~IDrawPhase() override = default;

    virtual void Draw(Frame & frame) = 0;
};
}  // namespace My
