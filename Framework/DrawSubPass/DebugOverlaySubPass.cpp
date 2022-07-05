#include "DebugOverlaySubPass.hpp"

#include "GraphicsManager.hpp"
#include "imgui.h"

using namespace My;
using namespace std;

DebugOverlaySubPass::~DebugOverlaySubPass() {
    for (auto& texture_debug_view : m_TextureViews) {
        m_pGraphicsManager->ReleaseTexture(texture_debug_view);
    }
}

void DebugOverlaySubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        size_t texture_view_index = 0;

        ImGui::Begin((const char*)u8"调试窗口");  // Create a debug window

        ImGui::Text(
            (const char*)u8"阴影贴图");

        for (int32_t i = 0; i < frame.frameContext.globalShadowMap.size; i++) {
            Texture2D texture_debug_view;
            if (texture_view_index >= m_TextureViews.size()) {
                m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.frameContext.globalShadowMap, i);
                m_TextureViews.push_back(texture_debug_view);
            } else {
                texture_debug_view = m_TextureViews[texture_view_index];
            }
            
            ImGui::Image((ImTextureID)texture_debug_view.handler, ImVec2(128, 128));
            ++texture_view_index;
        }

        for (int32_t i = 0; i < frame.frameContext.shadowMap.size; i++) {
            Texture2D texture_debug_view;
            if (texture_view_index >= m_TextureViews.size()) {
                m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.frameContext.shadowMap, i);
                m_TextureViews.push_back(texture_debug_view);
            } else {
                texture_debug_view = m_TextureViews[texture_view_index];
            }

            ImGui::Image((ImTextureID)texture_debug_view.handler, ImVec2(128, 128));
            ++texture_view_index;
        }

        for (int32_t i = 0; i < frame.frameContext.cubeShadowMap.size; i++) {
            Texture2D texture_debug_view;
            if (texture_view_index >= m_TextureViews.size()) {
                m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.frameContext.cubeShadowMap, i);
                m_TextureViews.push_back(texture_debug_view);
            } else {
                texture_debug_view = m_TextureViews[texture_view_index];
            }

            ImGui::Image((ImTextureID)texture_debug_view.handler, ImVec2(128, 128));
            ++texture_view_index;
        }

        // Draw Skybox
        {
            ImGui::Text(
                (const char*)u8"天空盒");

            auto start_index = texture_view_index;
            for (int32_t i = 0; i < 6; i++) {
                Texture2D texture_debug_view;

                if (texture_view_index >= m_TextureViews.size()) {
                    m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i);
                    m_TextureViews.push_back(texture_debug_view);
                } 

                texture_debug_view = m_TextureViews[start_index + i];
                ImGui::Image((ImTextureID)texture_debug_view.handler, ImVec2(64, 64));
                if (!(i % 2)) ImGui::SameLine();
                ++texture_view_index;
            }

        }

        // BRDF LUT
        ImGui::Text(
            (const char*)u8"BRDF LUT");

        ImGui::Image((ImTextureID)frame.brdfLUT.handler, ImVec2(128, 128));

        ImGui::End(); // finish the debug window
    }
}
