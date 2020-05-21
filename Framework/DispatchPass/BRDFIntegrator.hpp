#pragma once
#include "IDispatchPass.hpp"

namespace My {
class BRDFIntegrator : _implements_ IDispatchPass {
   public:
    ~BRDFIntegrator() override = default;
    void Dispatch(Frame& frame) final;
};
}  // namespace My