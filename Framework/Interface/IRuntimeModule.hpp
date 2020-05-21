#pragma once

#include <config.h>

#include "Interface.hpp"

namespace My {
_Interface_ IRuntimeModule {
   public:
    virtual ~IRuntimeModule() = default;

    virtual int Initialize() = 0;
    virtual void Finalize() = 0;

    virtual void Tick() = 0;

#ifdef DEBUG
    virtual void DrawDebugInfo(){};
#endif
};

}  // namespace My
