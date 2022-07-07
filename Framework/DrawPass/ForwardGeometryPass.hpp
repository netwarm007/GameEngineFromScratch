#pragma once
#include "BaseDrawPass.hpp"
#include "DebugOverlaySubPass.hpp"
#include "GeometrySubPass.hpp"
#include "GuiSubPass.hpp"
#include "SkyBoxSubPass.hpp"

namespace My {
class ForwardGeometryPass : public BaseDrawPass {
   public:
    ForwardGeometryPass(IGraphicsManager* pGfxMgr,
                        IPipelineStateManager* pPipeMgr)
        : BaseDrawPass(pGfxMgr, pPipeMgr) {
        m_DrawSubPasses.push_back(std::make_shared<GeometrySubPass>(
            m_pGraphicsManager, m_pPipelineStateManager));
        m_DrawSubPasses.push_back(std::make_shared<SkyBoxSubPass>(
            m_pGraphicsManager, m_pPipelineStateManager));
#if !defined(OS_WEBASSEMBLY)
        // m_DrawSubPasses.push_back(std::make_shared<TerrainSubPass>(m_pGraphicsManager));
#endif
        m_bClearRT = true;
    }
};
}  // namespace My
