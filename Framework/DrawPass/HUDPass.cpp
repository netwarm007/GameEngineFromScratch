#include "HUDPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;
using namespace std;

void HUDPass::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Copy);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

#ifdef DEBUG
    // Draw Shadow Maps
    float top = 0.95f;
    float left = 0.70f;

    for (uint32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawTextureOverlay(frame.frameContext.globalShadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    for (uint32_t i = 0; i < frame.frameContext.shadowMapCount; i++)
    {
        g_pGraphicsManager->DrawTextureOverlay(frame.frameContext.shadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::CopyCube);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    for (uint32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawCubeMapOverlay(frame.frameContext.cubeShadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::CopyCube2);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    // SkyBox
    g_pGraphicsManager->DrawCubeMapOverlay(frame.frameContext.skybox, left, top, 0.25f, 0.25f);

#endif
}
