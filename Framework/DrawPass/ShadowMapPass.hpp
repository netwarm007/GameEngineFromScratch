#pragma once
#include "BasePass.hpp"

namespace My {
    class ShadowMapPass: public BasePass
    {
    public:
        ~ShadowMapPass() = default;
        void Draw(Frame& frame) final;
    };
}
