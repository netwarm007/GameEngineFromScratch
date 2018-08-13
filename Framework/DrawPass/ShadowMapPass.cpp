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
    g_pGraphicsManager->SetPerFrameConstants(frame.frameContext);

    if (frame.shadowMap != -1)
        g_pGraphicsManager->DestroyShadowMap(frame.shadowMap);

    frame.shadowMapCount = 0;

    // count shadow map
    vector<decltype(frame.frameContext.m_lights)::iterator> lights_cast_shadow;

    for (auto it = frame.frameContext.m_lights.begin(); it != frame.frameContext.m_lights.end(); it++)
    {
        if (it->m_lightCastShadow)
        {
            frame.shadowMapCount++;
            lights_cast_shadow.push_back(it);
        }
    }

    // generate shadow map array
    frame.shadowMap = g_pGraphicsManager->GenerateShadowMapArray(frame.shadowMapCount);

    uint32_t shadowmap_index = 0;

    for (auto it : lights_cast_shadow)
    {
        // update shadow map
        g_pGraphicsManager->BeginShadowMap(*it, frame.shadowMap, shadowmap_index);

        for (auto dbc : frame.batchContexts)
        {
            g_pGraphicsManager->DrawBatchDepthOnly(*dbc);
        }

        g_pGraphicsManager->EndShadowMap(frame.shadowMap, shadowmap_index);

        it->m_lightShadowMapIndex = shadowmap_index++;
    }
}