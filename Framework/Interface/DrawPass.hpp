#pragma once
#include <iostream>
#include "Interface.hpp"

namespace My {
    Interface DrawPass
    {
    public:
        DrawPass() = default;
        virtual ~DrawPass() {};

        virtual void Draw(void) = 0;
    };
}
