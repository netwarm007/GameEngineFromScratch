#pragma once
#include "FrameStructure.hpp"
#include "Interface.hpp"

namespace My {
    Interface IDrawPass
    {
    public:
        IDrawPass() = default;
        virtual ~IDrawPass() = default;

        virtual void Draw(Frame& frame) = 0;
    };
}
