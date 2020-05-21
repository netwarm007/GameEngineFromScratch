#pragma once
#include <iostream>

#include "Interface.hpp"

namespace My {
_Interface_ IPhase {
   public:
    IPhase() = default;
    virtual ~IPhase() = default;

    virtual void BeginPhase(){};
    virtual void EndPhase(){};
};
}  // namespace My
