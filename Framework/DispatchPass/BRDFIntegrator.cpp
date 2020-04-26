#include "BRDFIntegrator.hpp"
#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;

void BRDFIntegrator::Dispatch()
{
    auto& pPipelineState = g_pPipelineStateManager->GetPipelineState("PBR BRDF CS");

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->SetPipelineState(pPipelineState);

    int32_t brdf_lut;
    const uint32_t width = 512u;
    const uint32_t height = 512u;
    const uint32_t depth = 1u;
    brdf_lut = g_pGraphicsManager->GenerateAndBindTextureForWrite("BRDF_LUT", 0, width, height);
    g_pGraphicsManager->Dispatch(width, height, depth);
}