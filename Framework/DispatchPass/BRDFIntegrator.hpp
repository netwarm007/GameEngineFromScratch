#pragma once
#include "IDispatchPass.hpp"

namespace My {
    class BRDFIntegrator : implements IDispatchPass
    {
    public:
        ~BRDFIntegrator() = default; 
        void Dispatch(void) final;
    };
}