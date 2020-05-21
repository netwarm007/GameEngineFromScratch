#pragma once
#include "BasePass.hpp"
#include "ForwardRenderPhase.hpp"
#include "HUDPhase.hpp"
#include "SkyBoxPhase.hpp"

namespace My {
class ForwardGeometryPass : public BasePass {
   public:
    ForwardGeometryPass() {
        m_DrawPhases.push_back(std::make_shared<ForwardRenderPhase>());
        m_DrawPhases.push_back(std::make_shared<SkyBoxPhase>());
        m_DrawPhases.push_back(std::make_shared<HUDPhase>());
    }

    ~ForwardGeometryPass() override = default;
};
}  // namespace My
