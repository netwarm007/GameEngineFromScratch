#include "DebugOverlaySubPass.hpp"

#include "GraphicsManager.hpp"
#include "imgui.h"

using namespace My;
using namespace std;

void DebugOverlaySubPass::Draw(Frame& frame) {
#ifdef DEBUG
    if (ImGui::GetCurrentContext()) {
        ImGui::Begin((const char*)u8"调试窗口");  // Create a debug window

        ImGui::Text(
            (const char*)u8"阴影贴图");  // Display some text (you can use
                                            // a format strings too)
        for (int32_t i = 0; i < frame.frameContext.globalShadowMapCount; i++) {
            ImGui::Image((ImTextureID)frame.frameContext.globalShadowMap, ImVec2(128, 128));
        }

        for (int32_t i = 0; i < frame.frameContext.shadowMapCount; i++) {
            ImGui::Image((ImTextureID)frame.frameContext.shadowMap, ImVec2(128, 128));
        }

        for (int32_t i = 0; i < frame.frameContext.cubeShadowMapCount; i++) {
            ImGui::Image((ImTextureID)frame.frameContext.cubeShadowMap, ImVec2(128, 128));
        }

        ImGui::Text(
            (const char*)u8"天空盒");  // Display some text (you can use
                                            // a format strings too)
        // Draw Skybox
        ImGui::Image((ImTextureID)frame.skybox, ImVec2(128, 128));

        // BRDF LUT
        auto brdf_lut = m_pGraphicsManager->GetTexture("BRDF_LUT");
        ImGui::Image((ImTextureID)brdf_lut, ImVec2(128, 128));

        ImGui::End(); // finish the debug window
    }
#endif
}
