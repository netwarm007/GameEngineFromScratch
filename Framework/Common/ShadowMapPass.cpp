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
            intptr_t shadowMap;

            if (frame.shadowMaps.find(light.m_lightGuid) == frame.shadowMaps.end())
            {
                // generate new shadow map
                shadowMap = g_pGraphicsManager->GenerateShadowMap(light);
                frame.shadowMaps[light.m_lightGuid] = shadowMap;
            }
            else
            {
                // reuse existing shadow map
                shadowMap = frame.shadowMaps[light.m_lightGuid];
            }

            // update shadow map
            g_pGraphicsManager->BeginShadowMap(light, shadowMap);

            for (auto dbc : frame.batchContexts)
            {
                g_pGraphicsManager->DrawBatchDepthOnly(*dbc);
            }

            g_pGraphicsManager->EndShadowMap(shadowMap);
        }
    }
}