#pragma once
#include <vector>

#include "IDrawPass.hpp"
#include "IDrawSubPass.hpp"
#include "IGraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

namespace My {
class BaseDrawPass : _implements_ IDrawPass {
   public:
    BaseDrawPass(IGraphicsManager* pGfxMgr, IPipelineStateManager* pPipeMgr)
        : m_pGraphicsManager(pGfxMgr), m_pPipelineStateManager(pPipeMgr) {}
    void BeginPass(Frame& frame) override { m_pGraphicsManager->BeginPass(frame); }
    void Draw(Frame& frame) override;
    void EndPass(Frame& frame) override { m_pGraphicsManager->EndPass(frame); }

   protected:
    BaseDrawPass() = default;

   protected:
    std::vector<std::shared_ptr<IDrawSubPass>> m_DrawSubPasses;
    IGraphicsManager* m_pGraphicsManager;
    IPipelineStateManager* m_pPipelineStateManager;
};
}  // namespace My
