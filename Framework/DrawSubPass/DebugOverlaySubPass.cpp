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
        for (int32_t i = 0; i < frame.frameContext.globalShadowMap.size; i++) {
            ImGui::Image((ImTextureID)frame.frameContext.globalShadowMap.handler, ImVec2(128, 128));
        }

        for (int32_t i = 0; i < frame.frameContext.shadowMap.size; i++) {
            ImGui::Image((ImTextureID)frame.frameContext.shadowMap.handler, ImVec2(128, 128));
        }

        for (int32_t i = 0; i < frame.frameContext.cubeShadowMap.size; i++) {
            ImGui::Image((ImTextureID)frame.frameContext.cubeShadowMap.handler, ImVec2(128, 128));
        }

        ImGui::Text(
            (const char*)u8"天空盒");  // Display some text (you can use
                                            // a format strings too)
        // Draw Skybox
        ImGui::Image((ImTextureID)frame.skybox.handler, ImVec2(128, 128));

        // BRDF LUT
        ImGui::Image((ImTextureID)frame.brdfLUT.handler, ImVec2(128, 128));

        ImGui::End(); // finish the debug window
    }
#endif
}
