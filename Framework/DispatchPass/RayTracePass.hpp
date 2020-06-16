#pragma once
#include "BaseDispatchPass.hpp"

namespace My {
class RayTracePass : public BaseDispatchPass {
    void Dispatch(Frame& frame) final;
};
}  // namespace My