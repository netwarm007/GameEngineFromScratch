#pragma once
#include "BaseDrawPass.hpp"
#include "DebugOverlaySubPass.hpp"
#include "GeometrySubPass.hpp"
#include "GuiSubPass.hpp"
#include "SkyBoxSubPass.hpp"
#include "TerrainSubPass.hpp"

namespace My {
class ForwardGeometryPass : public BaseDrawPass {
   public:
    ForwardGeometryPass() {
        m_DrawSubPasses.push_back(std::make_shared<GeometrySubPass>());
        m_DrawSubPasses.push_back(std::make_shared<SkyBoxSubPass>());
#if !defined(OS_WEBASSEMBLY)
        // m_DrawSubPasses.push_back(std::make_shared<TerrainSubPass>());
#endif
        m_DrawSubPasses.push_back(std::make_shared<DebugOverlaySubPass>());
        m_DrawSubPasses.push_back(std::make_shared<GuiSubPass>());
    }
};
}  // namespace My
