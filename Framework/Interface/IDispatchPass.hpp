#pragma once
#include "IPass.hpp"

namespace My {
    Interface IDispatchPass : public IPass
    {
    public:
        IDispatchPass() = default;
        virtual ~IDispatchPass() {};

        virtual void Dispatch(void) = 0;
    };
}
