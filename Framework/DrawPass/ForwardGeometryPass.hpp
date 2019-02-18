#pragma once
#include "BasePass.hpp"
#include "ForwardRenderPhase.hpp"
#include "SkyBoxPhase.hpp"
#include "TerrainPhase.hpp"
#include "HUDPhase.hpp"

namespace My {
    class ForwardGeometryPass: public BasePass
    {
    public:
        ForwardGeometryPass()
        {
            m_DrawPhases.push_back(std::make_shared<ForwardRenderPhase>());
            m_DrawPhases.push_back(std::make_shared<SkyBoxPhase>());
#if !defined(OS_WEBASSEMBLY)
            m_DrawPhases.push_back(std::make_shared<TerrainPhase>());
#endif
            m_DrawPhases.push_back(std::make_shared<HUDPhase>());
        }

        ~ForwardGeometryPass() = default;
    };
}
