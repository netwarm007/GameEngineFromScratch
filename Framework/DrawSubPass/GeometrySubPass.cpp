#include "GeometrySubPass.hpp"

#include "GraphicsManager.hpp"

using namespace My;
using namespace std;

void GeometrySubPass::Draw(Frame& frame) {
    auto& pPipelineState = m_pPipelineStateManager->GetPipelineState("PBR");

    // Set the color shader as the current shader program and set the matrices
    // that it will use for rendering.
    m_pGraphicsManager->SetPipelineState(pPipelineState, frame);
    m_pGraphicsManager->SetShadowMaps(frame);
    m_pGraphicsManager->DrawBatch(frame);
}
