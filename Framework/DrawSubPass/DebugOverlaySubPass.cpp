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

static void TexturePreviewer(const TextureBase& texture) {
    constexpr uint32_t texture_preview_width = 64, texture_preview_height = 64;
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::Image((ImTextureID)texture.handler, ImVec2(texture_preview_width, texture_preview_height));
    if (ImGui::IsItemHovered())
    {
        float focus_sz = 32.0f;
        float focus_x = io.MousePos.x - pos.x - focus_sz * 0.5f;
        if (focus_x < 0.0f)
            focus_x = 0.0f;
        else if (focus_x > texture_preview_width - focus_sz)
            focus_x = texture_preview_width - focus_sz;
        float focus_y = io.MousePos.y - pos.y - focus_sz * 0.5f;
        if (focus_y < 0.0f)
            focus_y = 0.0f;
        else if (focus_y > texture_preview_height - focus_sz)
        focus_y = texture_preview_height - focus_sz;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(pos.x + focus_x, pos.y + focus_y), ImVec2(pos.x + focus_x + focus_sz, pos.y + focus_y + focus_sz), ImColor(255, 255, 0, 100));
        ImGui::BeginTooltip();
        ImVec2 uv0 = ImVec2((focus_x) / texture_preview_width, (focus_y) / texture_preview_height);
        ImVec2 uv1 = ImVec2((focus_x + focus_sz) / texture_preview_width, (focus_y + focus_sz) / texture_preview_height);
        ImGui::Text("UV(min): (%.2f, %.2f)", uv0.x, uv0.y);
        ImGui::Text("UV(max): (%.2f, %.2f)", uv1.x, uv1.y);
        ImGui::Image((ImTextureID)texture.handler, ImVec2(128, 128), uv0, uv1, ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
        ImGui::EndTooltip();
    }
}

void DebugOverlaySubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        size_t texture_view_index = 0;

        ImGui::Begin((const char*)u8"调试窗口");  // Create a debug window

        if (ImGui::CollapsingHeader((const char*)u8"全局贴图", true)) {
            ImGui::Text(
                (const char*)u8"阴影贴图");

            for (int32_t i = 0; i < frame.frameContext.globalShadowMap.size; i++) {
                Texture2D texture_debug_view;
                if (texture_view_index >= m_TextureViews.size()) {
                    m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.frameContext.globalShadowMap, i, 0);
                    m_TextureViews.push_back(texture_debug_view);
                } else {
                    texture_debug_view = m_TextureViews[texture_view_index];
                }
                
                TexturePreviewer(texture_debug_view);
                ImGui::SameLine();
                ++texture_view_index;
            }

            ImGui::Spacing();

            for (int32_t i = 0; i < frame.frameContext.shadowMap.size; i++) {
                Texture2D texture_debug_view;
                if (texture_view_index >= m_TextureViews.size()) {
                    m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.frameContext.shadowMap, i, 0);
                    m_TextureViews.push_back(texture_debug_view);
                } else {
                    texture_debug_view = m_TextureViews[texture_view_index];
                }

                TexturePreviewer(texture_debug_view);
                ImGui::SameLine();
                ++texture_view_index;
            }

            ImGui::Spacing();

            for (int32_t i = 0; i < frame.frameContext.cubeShadowMap.size; i++) {
                Texture2D texture_debug_view;
                if (texture_view_index >= m_TextureViews.size()) {
                    m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.frameContext.cubeShadowMap, i, 0);
                    m_TextureViews.push_back(texture_debug_view);
                } else {
                    texture_debug_view = m_TextureViews[texture_view_index];
                }

                TexturePreviewer(texture_debug_view);
                ImGui::SameLine();
                ++texture_view_index;
            }

            ImGui::Separator();

            // Draw Skybox
            {
                ImGui::Text(
                    (const char*)u8"天空盒");

                auto start_index = texture_view_index;

                for (int32_t i = 0; i < 6; i++) {
                    Texture2D texture_debug_view;

                    if (texture_view_index >= m_TextureViews.size()) {
                        m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i, 0);
                        m_TextureViews.push_back(texture_debug_view);
                    } 

                    ++texture_view_index;
                }

                ImGui::BeginTable("skybox", 4);
                // +Y
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 2]);
                // -X
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                TexturePreviewer(m_TextureViews[start_index + 1]);
                // +Z
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 4]);
                // +X
                ImGui::TableSetColumnIndex(2);
                TexturePreviewer(m_TextureViews[start_index + 0]);
                // -Z
                ImGui::TableSetColumnIndex(3);
                TexturePreviewer(m_TextureViews[start_index + 5]);
                // -Y
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 3]);

                ImGui::EndTable();
            }

            ImGui::Separator();

            // Draw Irradiance
            {
                ImGui::Text(
                    (const char*)u8"辐照度");

                auto start_index = texture_view_index;

                for (int32_t i = 0; i < 6; i++) {
                    Texture2D texture_debug_view;

                    if (texture_view_index >= m_TextureViews.size()) {
                        m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i, 1);
                        m_TextureViews.push_back(texture_debug_view);
                    } 

                    ++texture_view_index;
                }

                ImGui::BeginTable("irradiance", 4);
                // +Y
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 2]);
                // -X
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                TexturePreviewer(m_TextureViews[start_index + 1]);
                // +Z
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 4]);
                // +X
                ImGui::TableSetColumnIndex(2);
                TexturePreviewer(m_TextureViews[start_index + 0]);
                // -Z
                ImGui::TableSetColumnIndex(3);
                TexturePreviewer(m_TextureViews[start_index + 5]);
                // -Y
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 3]);

                ImGui::EndTable();
            }

            // Draw Radiance
            {
                ImGui::Text(
                    (const char*)u8"辐射度");

                auto start_index = texture_view_index;

                for (int32_t i = 6; i < 12; i++) {
                    Texture2D texture_debug_view;

                    if (texture_view_index >= m_TextureViews.size()) {
                        m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i, 0);
                        m_TextureViews.push_back(texture_debug_view);
                    } 

                    ++texture_view_index;
                }

                ImGui::BeginTable("radiance", 4);
                // +Y
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 2]);
                // -X
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                TexturePreviewer(m_TextureViews[start_index + 1]);
                // +Z
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 4]);
                // +X
                ImGui::TableSetColumnIndex(2);
                TexturePreviewer(m_TextureViews[start_index + 0]);
                // -Z
                ImGui::TableSetColumnIndex(3);
                TexturePreviewer(m_TextureViews[start_index + 5]);
                // -Y
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                TexturePreviewer(m_TextureViews[start_index + 3]);

                ImGui::EndTable();
            }

            // BRDF LUT
            ImGui::Text(
                (const char*)u8"BRDF LUT");

            TexturePreviewer(frame.brdfLUT);
        }
        ImGui::End(); // finish the debug window
    }
}
