#include "DebugOverlaySubPass.hpp"

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;
using namespace std;

void DebugOverlaySubPass::Draw(Frame& frame) {
#ifdef DEBUG
    // Draw Shadow Maps
    m_pGraphicsManager->SetPipelineState(
        m_pPipelineStateManager->GetPipelineState("Texture Array Debug Output"),
        frame);

    float top = 0.95f;
    float left = 0.86f;

    for (int32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++) {
        m_pGraphicsManager->DrawTextureArrayOverlay(
            frame.frameContext.globalShadowMap, static_cast<float>(i), left,
            top, 0.12f, 0.12f);
        top -= 0.15f;
    }

    for (int32_t i = 0; i < frame.frameContext.shadowMapCount; i++) {
        m_pGraphicsManager->DrawTextureArrayOverlay(
            frame.frameContext.shadowMap, static_cast<float>(i), left, top,
            0.12f, 0.12f);
        top -= 0.15f;
    }

    m_pGraphicsManager->SetPipelineState(
        m_pPipelineStateManager->GetPipelineState("CubeMap Array Debug Output"),
        frame);

    for (int32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++) {
        m_pGraphicsManager->DrawCubeMapArrayOverlay(
            frame.frameContext.cubeShadowMap, static_cast<float>(i), left, top,
            0.12f, 0.12f, 0.0f);
        top -= 0.15f;
    }

    // Draw Skybox
    m_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 0.0f, left, top,
                                                0.12f, 0.12f, 0.0f);
    top -= 0.15f;

    // SkyBox Irradiance
    m_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 0.0f, left, top,
                                                0.12f, 0.12f, 1.0f);
    top -= 0.15f;

    // SkyBox Radiance
    m_pGraphicsManager->DrawCubeMapArrayOverlay(frame.skybox, 1.0f, left, top,
                                                0.12f, 0.12f, 1.0f);
    top -= 0.15f;

    // BRDF LUT
    m_pGraphicsManager->SetPipelineState(
        m_pPipelineStateManager->GetPipelineState("Texture Debug Output"),
        frame);

    auto brdf_lut = m_pGraphicsManager->GetTexture("BRDF_LUT");
    m_pGraphicsManager->DrawTextureOverlay(brdf_lut, left, top, 0.12f, 0.12f);
    top -= 0.15f;

    // auto raytrace = m_pGraphicsManager->GetTexture("RAYTRACE");
    // m_pGraphicsManager->DrawTextureOverlay(raytrace, left, top, 0.12f, 0.12f);
#endif
}
