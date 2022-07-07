#include "BaseDrawPass.hpp"

using namespace My;

void BaseDrawPass::Draw(Frame& frame) {
    for (const auto& pSubPass : m_DrawSubPasses) {
        pSubPass->BeginSubPass();
        pSubPass->Draw(frame);
        pSubPass->EndSubPass();
    }
}
    
void BaseDrawPass::BeginPass(Frame& frame) { 
    frame.renderToTexture = m_bRenderToTexture;

    m_pGraphicsManager->BeginPass(frame); 
}
