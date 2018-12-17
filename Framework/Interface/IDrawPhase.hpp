#pragma once
#include "IPhase.hpp"
#include "FrameStructure.hpp"

namespace My {
    Interface IDrawPhase : public IPhase
    {
    public:
        IDrawPhase() = default;
        virtual ~IDrawPhase() {};

        virtual void Draw(Frame& frame) = 0;
    };
}
