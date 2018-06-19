#include "ShadowMapPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace std;
using namespace My;

void ShadowMapPass::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::ShadowMap);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    for (auto light : frame.frameContext.m_lights)
    {
        if (light.m_bCastShadow)
        {
            // generate shadow map here
            intptr_t shadowMap = g_pGraphicsManager->GenerateShadowMap(light);
            frame.shadowMaps[light.m_lightGuid] = shadowMap;

            for (auto dbc : frame.batchContexts)
            {
                g_pGraphicsManager->DrawBatchDepthOnly(*dbc);
            }

            g_pGraphicsManager->FinishShadowMap(light);
        }
    }
}