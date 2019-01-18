#include "HUDPhase.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;
using namespace std;

void HUDPhase::Draw(Frame& frame)
{
#ifdef DEBUG
    // Draw Shadow Maps
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::CopyArray);

    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    float top = 0.95f;
    float left = 0.70f;

    for (int32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawTextureArrayOverlay(frame.frameContext.globalShadowMap, static_cast<float>(i), left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    for (int32_t i = 0; i < frame.frameContext.shadowMapCount; i++)
    {
        g_pGraphicsManager->DrawTextureArrayOverlay(frame.frameContext.shadowMap, static_cast<float>(i), left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::CopyCubeArray);

    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    for (int32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.cubeShadowMap, static_cast<float>(i), left, top, 0.25f, 0.25f, 0.0f);
        top -= 0.30f;
    }

    // Draw Skybox
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    // SkyBox
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.skybox, 0.0f, left, top, 0.25f, 0.25f, 0.0f);
    top -= 0.30f;

    // SkyBox Irradiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.skybox, 0.0f, left, top, 0.25f, 0.25f, 1.0f);
    top -= 0.30f;

    // SkyBox Radiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.frameContext.skybox, 1.0f, left, top, 0.25f, 0.25f, 1.0f);
    top -= 0.30f;

    // BRDF LUT
    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Copy);
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    auto brdf_lut = g_pGraphicsManager->GetTexture("BRDF_LUT");
    g_pGraphicsManager->DrawTextureOverlay(brdf_lut, left, top, 0.25f, 0.25f);
#endif
}
