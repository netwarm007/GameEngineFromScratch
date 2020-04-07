#pragma once
#include "Interface.hpp"
#include <iostream>

namespace My {
    Interface IPhase
    {
    public:
        IPhase() = default;
        virtual ~IPhase() = default;

        virtual void BeginPhase() {};
        virtual void EndPhase() {};
    };
}
