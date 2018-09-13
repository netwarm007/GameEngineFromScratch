#include "ShadowMapPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace std;
using namespace My;

void ShadowMapPass::Draw(Frame& frame)
{
    if (frame.frameContext.globalShadowMap != -1)
    {
        g_pGraphicsManager->DestroyShadowMap(frame.frameContext.globalShadowMap);
        frame.frameContext.globalShadowMapCount = 0;
    }

    if (frame.frameContext.shadowMap != -1)
    {
        g_pGraphicsManager->DestroyShadowMap(frame.frameContext.shadowMap);
        frame.frameContext.shadowMapCount = 0;
    }

    if (frame.frameContext.cubeShadowMap != -1)
    {
        g_pGraphicsManager->DestroyShadowMap(frame.frameContext.cubeShadowMap);
        frame.frameContext.cubeShadowMapCount = 0;
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
                    frame.frameContext.cubeShadowMapCount++;
                    break;
                case LightType::Spot:
                    frame.frameContext.shadowMapCount++;
                    break;
                case LightType::Area:
                    frame.frameContext.shadowMapCount++;
                    break;
                case LightType::Infinity:
                    frame.frameContext.globalShadowMapCount++;
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
    frame.frameContext.shadowMap = g_pGraphicsManager->GenerateShadowMapArray(kShadowMapWidth, kShadowMapHeight, frame.frameContext.shadowMapCount);

    // generate global shadow map array
    frame.frameContext.globalShadowMap = g_pGraphicsManager->GenerateShadowMapArray(kGlobalShadowMapWidth, kGlobalShadowMapHeight, frame.frameContext.globalShadowMapCount);

    // generate cube shadow map array
    frame.frameContext.cubeShadowMap = g_pGraphicsManager->GenerateCubeShadowMapArray(kCubeShadowMapWidth, kCubeShadowMapHeight, frame.frameContext.cubeShadowMapCount);

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
                shadowmap = frame.frameContext.cubeShadowMap;
                width = kCubeShadowMapWidth;
                height = kCubeShadowMapHeight;
                it->m_lightShadowMapIndex = cube_shadowmap_index++;
                break;
            case LightType::Spot:
                shadowmap = frame.frameContext.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it->m_lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Area:
                shadowmap = frame.frameContext.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it->m_lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Infinity:
                shadowmap = frame.frameContext.globalShadowMap;
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

        g_pGraphicsManager->BeginShadowMap(*it, shadowmap, 
            width, height, it->m_lightShadowMapIndex);

        for (auto dbc : frame.batchContexts)
        {
            g_pGraphicsManager->DrawBatchDepthOnly(*dbc);
        }

        g_pGraphicsManager->EndShadowMap(shadowmap, it->m_lightShadowMapIndex);
    }

    assert(shadowmap_index == frame.frameContext.shadowMapCount);
    assert(global_shadowmap_index == frame.frameContext.globalShadowMapCount);
    assert(cube_shadowmap_index == frame.frameContext.cubeShadowMapCount);
}