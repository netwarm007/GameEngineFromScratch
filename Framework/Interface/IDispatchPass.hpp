#pragma once
#include "Interface.hpp"

namespace My {
    Interface IDispatchPass
    {
    public:
        IDispatchPass() = default;
        virtual ~IDispatchPass() = default;

        virtual void Dispatch() = 0;
    };
}
