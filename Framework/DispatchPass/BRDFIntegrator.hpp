#pragma once
#include "BaseDispatchPass.hpp"

namespace My {
class BRDFIntegrator : public BaseDispatchPass {
   public:
    void Dispatch(Frame& frame) final;
};
}  // namespace My