#include "BRDFIntegrator.hpp"

#include "GraphicsManager.hpp"

using namespace My;

void BRDFIntegrator::Dispatch(Frame& frame) {
    auto& pPipelineState =
        m_pPipelineStateManager->GetPipelineState("PBR BRDF CS");

    // Set the color shader as the current shader program and set the matrices
    // that it will use for rendering.
    m_pGraphicsManager->SetPipelineState(pPipelineState, frame);

    const uint32_t width = 512u;
    const uint32_t height = 512u;
    const uint32_t depth = 1u;
    if (!frame.brdfLUT.handler) {
        frame.brdfLUT.width = width;
        frame.brdfLUT.height = height;
        m_pGraphicsManager->GenerateTextureForWrite(frame.brdfLUT);
    }
    m_pGraphicsManager->BindTextureForWrite(frame.brdfLUT, 0);
    m_pGraphicsManager->Dispatch(width, height, depth);
}