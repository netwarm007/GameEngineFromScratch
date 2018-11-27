#include "BRDFIntegrator.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;

void BRDFIntegrator::Dispatch()
{
    if(g_pGraphicsManager->CheckCapability(RHICapability::COMPUTE_SHADER))
    {
        auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::PbrBrdf);

        // Set the color shader as the current shader program and set the matrices that it will use for rendering.
        g_pGraphicsManager->UseShaderProgram(shaderProgram);

        int32_t brdf_lut;
        const uint32_t width = 512u;
        const uint32_t height = 512u;
        const uint32_t depth = 1u;
        brdf_lut = g_pGraphicsManager->GenerateAndBindTextureForWrite("BRDF_LUT", width, height);
        g_pGraphicsManager->Dispatch(width, height, depth);
    }
    else // using pixel shader instead
    {
        auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::PbrBrdfPs);

        // Set the color shader as the current shader program and set the matrices that it will use for rendering.
        g_pGraphicsManager->UseShaderProgram(shaderProgram);

        int32_t brdf_lut;
        int32_t brdf_context;
        const uint32_t width = 512u;
        const uint32_t height = 512u;
        brdf_lut = g_pGraphicsManager->GenerateTexture("BRDF_LUT", width, height);
        g_pGraphicsManager->BeginRenderToTexture(brdf_context, brdf_lut, width, height);
        g_pGraphicsManager->DrawFullScreenQuad();
        g_pGraphicsManager->EndRenderToTexture(brdf_context);
    }
}