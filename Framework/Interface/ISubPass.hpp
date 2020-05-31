#pragma once
#include <iostream>

#include "Interface.hpp"

namespace My {
_Interface_ ISubPass {
   public:
    virtual void BeginSubPass() = 0;
    virtual void EndSubPass() = 0;
};
}  // namespace My
