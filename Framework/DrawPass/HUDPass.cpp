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

    for (uint32_t i = 0; i < frame.shadowMapCount; i++)
    {
        g_pGraphicsManager->DrawOverlay(frame.shadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    top = 0.95f;
    left = 0.40f;

    for (uint32_t i = 0; i < frame.globalShadowMapCount; i++)
    {
        g_pGraphicsManager->DrawOverlay(frame.globalShadowMap, i, left, top, 0.25f, 0.25f);
        top -= 0.30f;
    }
#endif
}
