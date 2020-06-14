#pragma once
#include "IDispatchPass.hpp"

namespace My {
class BaseDispatchPass : _implements_ IDispatchPass {
   public:
    void BeginPass() override;
    void EndPass() override;

   protected:
    BaseDispatchPass() = default;
};
}  // namespace My
