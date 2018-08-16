#include "ShadowMapPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace std;
using namespace My;

void ShadowMapPass::Draw(Frame& frame)
{
    if (frame.globalShadowMap != -1)
    {
        g_pGraphicsManager->DestroyShadowMap(frame.globalShadowMap);
        frame.globalShadowMapCount = 0;
    }

    if (frame.shadowMap != -1)
    {
        g_pGraphicsManager->DestroyShadowMap(frame.shadowMap);
        frame.shadowMapCount = 0;
    }

    if (frame.cubeShadowMap != -1)
    {
        g_pGraphicsManager->DestroyShadowMap(frame.cubeShadowMap);
        frame.cubeShadowMapCount = 0;
    }

    // count shadow map
    vector<decltype(frame.frameContext.m_lights)::iterator> lights_cast_shadow;

    for (auto it = frame.frameContext.m_lights.begin(); it != frame.frameContext.m_lights.end(); it++)
    {
        if (it->m_lightCastShadow)
        {
            switch (it->m_lightType)
            {
                case LightType::Omni:
                    frame.cubeShadowMapCount++;
                    break;
                case LightType::Spot:
                    frame.shadowMapCount++;
                    break;
                case LightType::Area:
                    frame.shadowMapCount++;
                    break;
                case LightType::Infinity:
                    frame.globalShadowMapCount++;
                    break;
                default:
                    assert(0);
            }

            lights_cast_shadow.push_back(it);
        }
    }

    const uint32_t kShadowMapWidth = 512; // normal shadow map
    const uint32_t kShadowMapHeight = 512; // normal shadow map
    const uint32_t kCubeShadowMapWidth = 512; // cube shadow map
    const uint32_t kCubeShadowMapHeight = 512; // cube shadow map
    const uint32_t kGlobalShadowMapWidth = 2048;  // shadow map for sun light
    const uint32_t kGlobalShadowMapHeight = 2048; // shadow map for sun light

    // generate shadow map array
    frame.shadowMap = g_pGraphicsManager->GenerateShadowMapArray(kShadowMapWidth, kShadowMapHeight, frame.shadowMapCount);

    // generate global shadow map array
    frame.globalShadowMap = g_pGraphicsManager->GenerateShadowMapArray(kGlobalShadowMapWidth, kGlobalShadowMapHeight, frame.globalShadowMapCount);

    // generate cube shadow map array
    frame.cubeShadowMap = g_pGraphicsManager->GenerateCubeShadowMapArray(kCubeShadowMapWidth, kCubeShadowMapHeight, frame.cubeShadowMapCount);

    uint32_t shadowmap_index = 0;
    uint32_t global_shadowmap_index = 0;
    uint32_t cube_shadowmap_index = 0;

    for (auto it : lights_cast_shadow)
    {
        intptr_t shadowmap;
        DefaultShaderIndex shader_index = DefaultShaderIndex::ShadowMap;
        int32_t width, height;

        switch (it->m_lightType)
        {
            case LightType::Omni:
                shader_index = DefaultShaderIndex::OmniShadowMap;
                shadowmap = frame.cubeShadowMap;
                width = kCubeShadowMapWidth;
                height = kCubeShadowMapHeight;
                it->m_lightShadowMapIndex = cube_shadowmap_index++;
                break;
            case LightType::Spot:
                shadowmap = frame.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it->m_lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Area:
                shadowmap = frame.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it->m_lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Infinity:
                shadowmap = frame.globalShadowMap;
                width = kGlobalShadowMapWidth;
                height = kGlobalShadowMapHeight;
                it->m_lightShadowMapIndex = global_shadowmap_index++;
                break;
            default:
                assert(0);
        }

        auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(shader_index);

        // Set the color shader as the current shader program and set the matrices that it will use for rendering.
        g_pGraphicsManager->UseShaderProgram(shaderProgram);
        g_pGraphicsManager->SetPerFrameConstants(frame.frameContext);

        g_pGraphicsManager->BeginShadowMap(*it, shadowmap, 
            width, height, it->m_lightShadowMapIndex);

        for (auto dbc : frame.batchContexts)
        {
            g_pGraphicsManager->DrawBatchDepthOnly(*dbc);
        }

        g_pGraphicsManager->EndShadowMap(shadowmap, it->m_lightShadowMapIndex);
    }

    assert(shadowmap_index == frame.shadowMapCount);
    assert(global_shadowmap_index == frame.globalShadowMapCount);
    assert(cube_shadowmap_index == frame.cubeShadowMapCount);
}