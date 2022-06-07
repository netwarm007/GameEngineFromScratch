#pragma once
#include "IDispatchPass.hpp"
#include "IGraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

namespace My {
class BaseDispatchPass : _implements_ IDispatchPass {
   public:
    BaseDispatchPass(IGraphicsManager* pGfxMgr, IPipelineStateManager* pPipeMgr)
        : m_pGraphicsManager(pGfxMgr), m_pPipelineStateManager(pPipeMgr) {}
    void BeginPass(Frame&) override;
    void EndPass(Frame&) override;

   protected:
    IGraphicsManager* m_pGraphicsManager;
    IPipelineStateManager* m_pPipelineStateManager;
};
}  // namespace My
