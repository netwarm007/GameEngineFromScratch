#pragma once
#include "FrameStructure.hpp"
#include "IPhase.hpp"

namespace My {
    Interface IDrawPhase : public IPhase
    {
    public:
        IDrawPhase() = default;
        ~IDrawPhase() override = default;;

        virtual void Draw(Frame& frame) = 0;
    };
}
