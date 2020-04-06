#pragma once
#include <iostream>
#include "Interface.hpp"

namespace My {
    Interface IPhase
    {
    public:
        IPhase() = default;
        virtual ~IPhase() {};

        virtual void BeginPhase() {};
        virtual void EndPhase() {};
    };
}
