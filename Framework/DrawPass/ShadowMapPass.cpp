#include "ShadowMapPass.hpp"

#include <utility>
#include <vector>

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace std;
using namespace My;

void ShadowMapPass::Draw(Frame& frame) {
    if (frame.frameContext.globalShadowMap != -1) {
        g_pGraphicsManager->DestroyShadowMap(
            frame.frameContext.globalShadowMap);
        frame.frameContext.globalShadowMapCount = 0;
    }

    if (frame.frameContext.shadowMap != -1) {
        g_pGraphicsManager->DestroyShadowMap(frame.frameContext.shadowMap);
        frame.frameContext.shadowMapCount = 0;
    }

    if (frame.frameContext.cubeShadowMap != -1) {
        g_pGraphicsManager->DestroyShadowMap(frame.frameContext.cubeShadowMap);
        frame.frameContext.cubeShadowMapCount = 0;
    }

    // count shadow map
    std::vector<std::pair<int32_t, Light*>> lights_cast_shadow;

    for (int32_t i = 0; i < frame.frameContext.numLights; i++) {
        auto& light = frame.lightInfo.lights[i];

        if (light.lightCastShadow) {
            switch (light.lightType) {
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

            lights_cast_shadow.emplace_back(i, &light);
        }
    }

    const uint32_t kShadowMapWidth = 512;          // normal shadow map
    const uint32_t kShadowMapHeight = 512;         // normal shadow map
    const uint32_t kCubeShadowMapWidth = 512;      // cube shadow map
    const uint32_t kCubeShadowMapHeight = 512;     // cube shadow map
    const uint32_t kGlobalShadowMapWidth = 2048;   // shadow map for sun light
    const uint32_t kGlobalShadowMapHeight = 2048;  // shadow map for sun light

    // generate shadow map array
    if (frame.frameContext.shadowMapCount) {
        frame.frameContext.shadowMap =
            g_pGraphicsManager->GenerateShadowMapArray(
                kShadowMapWidth, kShadowMapHeight,
                frame.frameContext.shadowMapCount);
    }

    // generate global shadow map array
    if (frame.frameContext.globalShadowMapCount) {
        frame.frameContext.globalShadowMap =
            g_pGraphicsManager->GenerateShadowMapArray(
                kGlobalShadowMapWidth, kGlobalShadowMapHeight,
                frame.frameContext.globalShadowMapCount);
    }

    // generate cube shadow map array
    if (frame.frameContext.cubeShadowMapCount) {
        frame.frameContext.cubeShadowMap =
            g_pGraphicsManager->GenerateCubeShadowMapArray(
                kCubeShadowMapWidth, kCubeShadowMapHeight,
                frame.frameContext.cubeShadowMapCount);
    }

    uint32_t shadowmap_index = 0;
    uint32_t global_shadowmap_index = 0;
    uint32_t cube_shadowmap_index = 0;

    for (auto it : lights_cast_shadow) {
        int32_t shadowmap;
        int32_t width, height;

        const char* pipelineStateName;

        switch (it.second->lightType) {
            case LightType::Omni:
                pipelineStateName = "Omni Light Shadow Map";
                shadowmap = frame.frameContext.cubeShadowMap;
                width = kCubeShadowMapWidth;
                height = kCubeShadowMapHeight;
                it.second->lightShadowMapIndex = cube_shadowmap_index++;
                break;
            case LightType::Spot:
                pipelineStateName = "Spot Light Shadow Map";
                shadowmap = frame.frameContext.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it.second->lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Area:
                pipelineStateName = "Area Light Shadow Map";
                shadowmap = frame.frameContext.shadowMap;
                width = kShadowMapWidth;
                height = kShadowMapHeight;
                it.second->lightShadowMapIndex = shadowmap_index++;
                break;
            case LightType::Infinity:
                pipelineStateName = "Sun Light Shadow Map";
                shadowmap = frame.frameContext.globalShadowMap;
                width = kGlobalShadowMapWidth;
                height = kGlobalShadowMapHeight;
                it.second->lightShadowMapIndex = global_shadowmap_index++;
                break;
            default:
                assert(0);
        }

        g_pGraphicsManager->BeginShadowMap(it.first, shadowmap, width, height,
                                           it.second->lightShadowMapIndex,
                                           frame);

        // Set the color shader as the current shader program and set the
        // matrices that it will use for rendering.
        auto& pPipelineState =
            g_pPipelineStateManager->GetPipelineState(pipelineStateName);
        g_pGraphicsManager->SetPipelineState(pPipelineState, frame);

        g_pGraphicsManager->DrawBatch(frame);

        g_pGraphicsManager->EndShadowMap(shadowmap,
                                         it.second->lightShadowMapIndex);
    }

    assert(shadowmap_index == frame.frameContext.shadowMapCount);
    assert(global_shadowmap_index == frame.frameContext.globalShadowMapCount);
    assert(cube_shadowmap_index == frame.frameContext.cubeShadowMapCount);
}
