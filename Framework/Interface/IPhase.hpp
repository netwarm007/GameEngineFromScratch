#pragma once
#include <iostream>
#include "Interface.hpp"

namespace My {
    Interface IPhase
    {
    public:
        IPhase() = default;
        virtual ~IPhase() = default;;

        virtual void BeginPhase() {};
        virtual void EndPhase() {};
    };
}
