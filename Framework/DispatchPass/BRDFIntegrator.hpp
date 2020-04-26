#pragma once
#include "IDispatchPass.hpp"

namespace My {
    class BRDFIntegrator : implements IDispatchPass
    {
    public:
        ~BRDFIntegrator() override = default; 
        void Dispatch() final;
    };
}