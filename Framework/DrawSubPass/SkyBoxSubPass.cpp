#include "SkyBoxSubPass.hpp"

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;

void SkyBoxSubPass::Draw(Frame& frame) {
    auto& pPipelineState = g_pPipelineStateManager->GetPipelineState("SkyBox");

    g_pGraphicsManager->SetPipelineState(pPipelineState, frame);

    g_pGraphicsManager->DrawSkyBox();
}
