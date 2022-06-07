#include "SkyBoxSubPass.hpp"

#include "GraphicsManager.hpp"

using namespace My;

void SkyBoxSubPass::Draw(Frame& frame) {
    auto& pPipelineState = m_pPipelineStateManager->GetPipelineState("SkyBox");

    m_pGraphicsManager->SetPipelineState(pPipelineState, frame);

    m_pGraphicsManager->DrawSkyBox(frame);
}
