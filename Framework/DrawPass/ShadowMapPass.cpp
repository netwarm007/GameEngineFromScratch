#include "ShadowMapPass.hpp"

#include <utility>
#include <vector>

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace std;
using namespace My;

void ShadowMapPass::Draw(Frame& frame) {
    uint32_t shadowmap_index = 0;
    uint32_t global_shadowmap_index = 0;
    uint32_t cube_shadowmap_index = 0;

    for (int32_t i = 0; i < frame.frameContext.numLights; i++) {
        auto& light = frame.lightInfo.lights[i];

        if (light.lightCastShadow) {
            int32_t shadowmap;
            int32_t width, height;

            const char* pipelineStateName;

            switch (light.lightType) {
                case LightType::Omni:
                    if (cube_shadowmap_index >=
                        GfxConfiguration::kMaxCubeShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Omni Light Shadow Map";
                    shadowmap = frame.frameContext.cubeShadowMap;
                    width = GfxConfiguration::kCubeShadowMapWidth;
                    height = GfxConfiguration::kCubeShadowMapHeight;
                    light.lightShadowMapIndex = cube_shadowmap_index++;
                    break;
                case LightType::Spot:
                    if (shadowmap_index >=
                        GfxConfiguration::kMaxShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Spot Light Shadow Map";
                    shadowmap = frame.frameContext.shadowMap;
                    width = GfxConfiguration::kShadowMapWidth;
                    height = GfxConfiguration::kShadowMapHeight;
                    light.lightShadowMapIndex = shadowmap_index++;
                    break;
                case LightType::Area:
                    if (shadowmap_index >=
                        GfxConfiguration::kMaxShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Area Light Shadow Map";
                    shadowmap = frame.frameContext.shadowMap;
                    width = GfxConfiguration::kShadowMapWidth;
                    height = GfxConfiguration::kShadowMapHeight;
                    light.lightShadowMapIndex = shadowmap_index++;
                    break;
                case LightType::Infinity:
                    if (global_shadowmap_index >=
                        GfxConfiguration::kMaxShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Sun Light Shadow Map";
                    shadowmap = frame.frameContext.globalShadowMap;
                    width = GfxConfiguration::kGlobalShadowMapWidth;
                    height = GfxConfiguration::kGlobalShadowMapHeight;
                    light.lightShadowMapIndex = global_shadowmap_index++;
                    break;
                default:
                    assert(0);
            }

            g_pGraphicsManager->BeginShadowMap(
                i, shadowmap, width, height, light.lightShadowMapIndex, frame);

            // Set the color shader as the current shader program and set the
            // matrices that it will use for rendering.
            auto& pPipelineState =
                g_pPipelineStateManager->GetPipelineState(pipelineStateName);
            g_pGraphicsManager->SetPipelineState(pPipelineState, frame);

            g_pGraphicsManager->DrawBatch(frame);

            g_pGraphicsManager->EndShadowMap(shadowmap,
                                             light.lightShadowMapIndex);
        }
    }

    frame.frameContext.globalShadowMapCount = global_shadowmap_index;
    frame.frameContext.cubeShadowMapCount = cube_shadowmap_index;
    frame.frameContext.shadowMapCount = shadowmap_index;
}
