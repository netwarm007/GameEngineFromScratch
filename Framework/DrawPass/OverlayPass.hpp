#pragma once
#include "BaseDrawPass.hpp"
#include "DebugOverlaySubPass.hpp"
#include "GuiSubPass.hpp"

namespace My {
class OverlayPass : public BaseDrawPass {
   public:
    OverlayPass(IGraphicsManager* pGfxMgr,
                        IPipelineStateManager* pPipeMgr)
        : BaseDrawPass(pGfxMgr, pPipeMgr) {
        m_DrawSubPasses.push_back(std::make_shared<DebugOverlaySubPass>(
            m_pGraphicsManager, m_pPipelineStateManager));
        m_DrawSubPasses.push_back(std::make_shared<GuiSubPass>(
            m_pGraphicsManager, m_pPipelineStateManager));
    }
};
}  // namespace My
