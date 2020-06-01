#pragma once
#include "IDispatchPass.hpp"

namespace My {
class BaseDispatchPass : _implements_ IDispatchPass {
   public:
    void BeginPass(Frame&) override {}
    void EndPass(Frame&) override {}

   protected:
    BaseDispatchPass() = default;
};
}  // namespace My
