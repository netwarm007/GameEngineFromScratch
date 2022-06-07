#pragma once
#include "BaseDispatchPass.hpp"

namespace My {
class BRDFIntegrator : public BaseDispatchPass {
   public:
    using BaseDispatchPass::BaseDispatchPass;
    void Dispatch(Frame& frame) final;
};
}  // namespace My