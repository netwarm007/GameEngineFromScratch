#include "RayTracePass.hpp"

using namespace My;

void RayTracePass::BeginPass() { 
    BaseDrawPass::BeginPass(); 
}

void RayTracePass::Draw(Frame& frame) {
    auto texture_raytrace = g_pGraphicsManager->GetTexture("Raytrace Texture");

    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("Texture Debug Output"),
        frame);

    g_pGraphicsManager->DrawTextureOverlay(texture_raytrace, -1.0f, 1.0f, 2.0f, 2.0f);
}

void RayTracePass::EndPass() { BaseDrawPass::EndPass(); }
