#pragma once
#include "BasePass.hpp"

namespace My {
    class ShadowMapPass: public BasePass
    {
    public:
        ~ShadowMapPass() override = default;

        void BeginPass() override {}
        void Draw(Frame& frame) final;
        void EndPass() override {}
    };
}
