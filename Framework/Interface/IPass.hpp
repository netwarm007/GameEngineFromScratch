#pragma once
#include <iostream>
#include "Interface.hpp"

namespace My {
    Interface IPass
    {
    public:
        IPass() = default;
        virtual ~IPass() {};

        virtual void BeginPass(void) {};
        virtual void EndPass(void) {};
    };
}
