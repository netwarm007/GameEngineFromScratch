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

        void Draw(Frame& frame) override;

    protected:
        BasePass() = default;

    protected:
        std::vector<std::shared_ptr<IDrawPhase>> m_DrawPhases;
    };
}
