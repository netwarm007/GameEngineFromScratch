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
    vector<Light*> lights_cast_shadow;

    for (int32_t i = 0; i < frame.frameContext.numLights; i++)
    {
        auto& light = frame.lightInfo.lights[i];
        
        if (light.lightCastShadow)
        {
            switch (light.lightType)
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

            lights_cast_shadow.push_back(&light);
        }
    }

    const uint32_t kShadowMapWidth = 512; // normal shadow map
    const uint32_t kShadowMapHeight = 512; // normal shadow map
    const uint32_t kCubeShadowMapWidth = 512; // cube shadow map
    const uint32_t kCubeShadowMapHeight = 512; // cube shadow map
    const uint32_t kGlobalShadowMapWidth = 2048;  // shadow map for sun light
    const uint32_t kGlobalShadowMapHeight = 2048; // shadow map for sun light

    // generate shadow map array
    if (frame.frameContext.shadowMapCount)
    {
        frame.frameContext.shadowMap = g_pGraphicsManager->GenerateShadowMapArray(kShadowMapWidth, kShadowMapHeight, frame.frameContext.shadowMapCount);
    }

    // generate global shadow map array
    if (frame.frameContext.globalShadowMapCount)
    {
        frame.frameContext.globalShadowMap = g_pGraphicsManager->GenerateShadowMapArray(kGlobalShadowMapWidth, kGlobalShadowMapHeight, frame.frameContext.globalShadowMapCount);
    }

    // generate cube shadow map array
    if (frame.frameContext.cubeShadowMapCount)
    {
        frame.frameContext.cubeShadowMap = g_pGraphicsManager->GenerateCubeShadowMapArray(kCubeShadowMapWidth, kCubeShadowMapHeight, frame.frameContext.cubeShadowMapCount);
    }

    uint32_t shadowmap_index = 0;
    uint32_t global_shadowmap_index = 0;
    uint32_t cube_shadowmap_index = 0;

    for (auto it : lights_cast_shadow)
    {
        int32_t shadowmap;
        DefaultShaderIndex shader_index = DefaultShaderIndex::ShadowMap;
        int32_t width, height;

        switch (it->lightType)
        {
            case LightType::Omni:
                shader_index = DefaultShaderIndex::OmniShadowMap;
                shadowmap = frame.frameContext.cubeShadowMap;
                width = kCubeShadowMapWidth;
                height = kCubeShadowMapHeight;
                it->lightShadowMapIndex = cube_shadowmap_index++;
                break;
            case LightType::Spot:
                shadowmap = frame.frameContext.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it->lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Area:
                shadowmap = frame.frameContext.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it->lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Infinity:
                shadowmap = frame.frameContext.globalShadowMap;
                width = kGlobalShadowMapWidth;
                height = kGlobalShadowMapHeight;
                it->lightShadowMapIndex = global_shadowmap_index++;
                break;
            default:
                assert(0);
        }

        auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(shader_index);

        // Set the color shader as the current shader program and set the matrices that it will use for rendering.
        g_pGraphicsManager->UseShaderProgram(shaderProgram);

        g_pGraphicsManager->BeginShadowMap(*it, shadowmap, 
            width, height, it->lightShadowMapIndex);

        g_pGraphicsManager->DrawBatch(frame.batchContexts);

        g_pGraphicsManager->EndShadowMap(shadowmap, it->lightShadowMapIndex);
    }

    assert(shadowmap_index == frame.frameContext.shadowMapCount);
    assert(global_shadowmap_index == frame.frameContext.globalShadowMapCount);
    assert(cube_shadowmap_index == frame.frameContext.cubeShadowMapCount);
}
