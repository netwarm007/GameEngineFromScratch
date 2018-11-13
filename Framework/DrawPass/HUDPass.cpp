#include "HUDPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;
using namespace std;

void HUDPass::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::DepthCopy);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

#if 0
    // Draw Shadow Maps
    float top = 0.95f;
    float left = 0.70f;

    for (uint32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawTextureArrayOverlay(frame.frameContext.globalShadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    for (uint32_t i = 0; i < frame.frameContext.shadowMapCount; i++)
    {
        g_pGraphicsManager->DrawTextureArrayOverlay(frame.frameContext.shadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::DepthCopyCube);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    for (uint32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.cubeShadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::CopyCube);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    // SkyBox
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.skybox, 0u, left, top, 0.25f, 0.25f, 0.0f);
    top -= 0.30f;

    // SkyBox Irradiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.skybox, 0u, left, top, 0.25f, 0.25f, 1.0f);
    top -= 0.30f;

    // SkyBox Radiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.skybox, 1u, left, top, 0.25f, 0.25f, 1.0f);
    top -= 0.30f;

    // BRDF LUT
    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Copy);
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    auto brdf_lut = g_pGraphicsManager->GetTexture("BRDF_LUT");
    g_pGraphicsManager->DrawTextureOverlay(brdf_lut, left, top, 0.25f, 0.25f);

#endif
}
