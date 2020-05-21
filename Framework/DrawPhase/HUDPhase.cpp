#include "HUDPhase.hpp"

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;
using namespace std;

void HUDPhase::Draw(Frame& frame) {
#ifdef DEBUG
    // Draw Shadow Maps
    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("Texture Array Debug Output"),
        frame);

    float top = 0.95f;
    float left = 0.70f;

    for (int32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++) {
        g_pGraphicsManager->DrawTextureArrayOverlay(
            frame.frameContext.globalShadowMap, static_cast<float>(i), left,
            top, 0.25f, 0.25f);
        top -= 0.30f;
    }

    for (int32_t i = 0; i < frame.frameContext.shadowMapCount; i++) {
        g_pGraphicsManager->DrawTextureArrayOverlay(
            frame.frameContext.shadowMap, static_cast<float>(i), left, top,
            0.25f, 0.25f);
        top -= 0.30f;
    }

    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("CubeMap Array Debug Output"),
        frame);

    for (int32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++) {
        g_pGraphicsManager->DrawCubeMapArrayOverlay(
            frame.frameContext.cubeShadowMap, static_cast<float>(i), left, top,
            0.25f, 0.25f, 0.0f);
        top -= 0.30f;
    }

    // Draw Skybox
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 0.0f, left, top,
                                                0.25f, 0.25f, 0.0f);
    top -= 0.30f;

    // SkyBox Irradiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 0.0f, left, top,
                                                0.25f, 0.25f, 1.0f);
    top -= 0.30f;

    // SkyBox Radiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 1.0f, left, top,
                                                0.25f, 0.25f, 1.0f);
    top -= 0.30f;

    // BRDF LUT
    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("Texture Debug Output"),
        frame);

    auto brdf_lut = g_pGraphicsManager->GetTexture("BRDF_LUT");
    g_pGraphicsManager->DrawTextureOverlay(brdf_lut, left, top, 0.25f, 0.25f);
#endif
}
