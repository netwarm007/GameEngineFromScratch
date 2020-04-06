#pragma once
#include "Interface.hpp"
#include "FrameStructure.hpp"

namespace My {
    Interface IDrawPass
    {
    public:
        IDrawPass() = default;
        virtual ~IDrawPass() = default;;

        virtual void Draw(Frame& frame) = 0;
    };
}
