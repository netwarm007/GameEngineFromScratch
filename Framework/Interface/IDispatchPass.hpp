#pragma once
#include "Interface.hpp"

namespace My {
    Interface IDispatchPass
    {
    public:
        IDispatchPass() = default;
        virtual ~IDispatchPass() {};

        virtual void Dispatch(void) = 0;
    };
}
