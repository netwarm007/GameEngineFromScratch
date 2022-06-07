#pragma once
#include "IDrawSubPass.hpp"
#include "IGraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

namespace My {
class BaseSubPass : _implements_ IDrawSubPass {
   public:
    BaseSubPass(IGraphicsManager* pGfxMgr, IPipelineStateManager* pPipeMgr)
        : m_pGraphicsManager(pGfxMgr), m_pPipelineStateManager(pPipeMgr) {}
    void BeginSubPass() override{};
    void EndSubPass() override{};

   protected:
    IGraphicsManager* m_pGraphicsManager;
    IPipelineStateManager* m_pPipelineStateManager;
};
}  // namespace My