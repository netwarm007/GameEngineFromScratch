#pragma once
#include <vector>
#include "IDrawPass.hpp"
#include "IDrawPhase.hpp"
#include "GraphicsManager.hpp"

namespace My {
    class BasePass : implements IDrawPass
    {
    public:
        ~BasePass() = default;
        void BeginPass(void) override;
        void EndPass(void) override;

        void Draw(Frame& frame) override;

    protected:
        BasePass() = default;

    protected:
        std::vector<std::shared_ptr<IDrawPhase>> m_DrawPhases;
    };
}
