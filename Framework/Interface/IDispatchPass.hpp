#pragma once
#include "Interface.hpp"
#include "FrameStructure.hpp"

namespace My {
    Interface IDispatchPass
    {
    public:
        IDispatchPass() = default;
        virtual ~IDispatchPass() = default;

        virtual void Dispatch(Frame& frame) = 0;
    };
}
