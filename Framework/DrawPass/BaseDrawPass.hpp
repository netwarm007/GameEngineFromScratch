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
    void BeginPass(Frame& frame) override;
    void Draw(Frame& frame) override;
    void EndPass(Frame& frame) override { m_pGraphicsManager->EndPass(frame); }
    void EnableRenderToTexture() { m_bRenderToTexture = true; }
    void DisableRenderToTexture() { m_bRenderToTexture = false; }

   protected:
    BaseDrawPass() = default;

   protected:
    std::vector<std::shared_ptr<IDrawSubPass>> m_DrawSubPasses;
    IGraphicsManager* m_pGraphicsManager;
    IPipelineStateManager* m_pPipelineStateManager;
    bool m_bRenderToTexture = false;
};
}  // namespace My
