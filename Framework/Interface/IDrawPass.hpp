#pragma once
#include "Interface.hpp"
#include "FrameStructure.hpp"

namespace My {
    Interface IDrawPass
    {
    public:
        IDrawPass() = default;
        virtual ~IDrawPass() {};

        virtual void Draw(Frame& frame) = 0;
    };
}
