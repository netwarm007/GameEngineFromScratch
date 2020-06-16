#include "DebugOverlaySubPass.hpp"

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;
using namespace std;

void DebugOverlaySubPass::Draw(Frame& frame) {
#ifdef DEBUG
    // Draw Shadow Maps
    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("Texture Array Debug Output"),
        frame);

    float top = 0.95f;
    float left = 0.86f;

    for (int32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++) {
        g_pGraphicsManager->DrawTextureArrayOverlay(
            frame.frameContext.globalShadowMap, static_cast<float>(i), left,
            top, 0.12f, 0.12f);
        top -= 0.15f;
    }

    for (int32_t i = 0; i < frame.frameContext.shadowMapCount; i++) {
        g_pGraphicsManager->DrawTextureArrayOverlay(
            frame.frameContext.shadowMap, static_cast<float>(i), left, top,
            0.12f, 0.12f);
        top -= 0.15f;
    }

    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("CubeMap Array Debug Output"),
        frame);

    for (int32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++) {
        g_pGraphicsManager->DrawCubeMapArrayOverlay(
            frame.frameContext.cubeShadowMap, static_cast<float>(i), left, top,
            0.12f, 0.12f, 0.0f);
        top -= 0.15f;
    }

    // Draw Skybox
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 0.0f, left, top,
                                                0.12f, 0.12f, 0.0f);
    top -= 0.15f;

    // SkyBox Irradiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 0.0f, left, top,
                                                0.12f, 0.12f, 1.0f);
    top -= 0.15f;

    // SkyBox Radiance
    g_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 1.0f, left, top,
                                                0.12f, 0.12f, 1.0f);
    top -= 0.15f;

    // BRDF LUT
    g_pGraphicsManager->SetPipelineState(
        g_pPipelineStateManager->GetPipelineState("Texture Debug Output"),
        frame);

    auto brdf_lut = g_pGraphicsManager->GetTexture("BRDF_LUT");
    g_pGraphicsManager->DrawTextureOverlay(brdf_lut, left, top, 0.12f, 0.12f);
    top -= 0.15f;

    auto raytrace = g_pGraphicsManager->GetTexture("RAYTRACE");
    g_pGraphicsManager->DrawTextureOverlay(raytrace, left, top, 0.12f, 0.12f);
#endif
}
