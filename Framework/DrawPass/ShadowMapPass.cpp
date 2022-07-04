#include "ShadowMapPass.hpp"

#include <utility>
#include <vector>

#include "GraphicsManager.hpp"

using namespace std;
using namespace My;

void ShadowMapPass::Draw(Frame& frame) {
    uint32_t shadowmap_index = 0;
    uint32_t global_shadowmap_index = 0;
    uint32_t cube_shadowmap_index = 0;

    for (int32_t i = 0; i < frame.frameContext.numLights; i++) {
        auto& light = frame.lightInfo.lights[i];

        if (light.lightCastShadow) {
            TextureBase* pShadowmap;

            const char* pipelineStateName;

            switch (light.lightType) {
                case LightType::Omni:
                    if (cube_shadowmap_index >=
                        GfxConfiguration::kMaxCubeShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Omni Light Shadow Map";
                    pShadowmap = &frame.frameContext.cubeShadowMap;
                    light.lightShadowMapIndex = cube_shadowmap_index++;
                    break;
                case LightType::Spot:
                    if (shadowmap_index >=
                        GfxConfiguration::kMaxShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Spot Light Shadow Map";
                    pShadowmap = &frame.frameContext.shadowMap;
                    light.lightShadowMapIndex = shadowmap_index++;
                    break;
                case LightType::Area:
                    if (shadowmap_index >=
                        GfxConfiguration::kMaxShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Area Light Shadow Map";
                    pShadowmap = &frame.frameContext.shadowMap;
                    light.lightShadowMapIndex = shadowmap_index++;
                    break;
                case LightType::Infinity:
                    if (global_shadowmap_index >=
                        GfxConfiguration::kMaxShadowMapCount) {
                        continue;
                    }
                    pipelineStateName = "Sun Light Shadow Map";
                    pShadowmap = &frame.frameContext.globalShadowMap;
                    light.lightShadowMapIndex = global_shadowmap_index++;
                    break;
                default:
                    assert(0);
            }

            m_pGraphicsManager->BeginShadowMap(
                i, pShadowmap, light.lightShadowMapIndex, frame);

            // Set the color shader as the current shader program and set the
            // matrices that it will use for rendering.
            auto& pPipelineState =
                m_pPipelineStateManager->GetPipelineState(pipelineStateName);
            m_pGraphicsManager->SetPipelineState(pPipelineState, frame);

            m_pGraphicsManager->DrawBatch(frame);

            m_pGraphicsManager->EndShadowMap(pShadowmap,
                                             light.lightShadowMapIndex, frame);
        }
    }

    frame.frameContext.globalShadowMap.size = global_shadowmap_index;
    frame.frameContext.cubeShadowMap.size = cube_shadowmap_index;
    frame.frameContext.shadowMap.size = shadowmap_index;
}
