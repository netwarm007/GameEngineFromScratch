#include "GuiSubPass.hpp"
#include "imgui/imgui.h"

#include <deque>

using namespace My;

GuiSubPass::~GuiSubPass() {
    for (auto& texture_debug_view : m_TextureViews) {
        m_pGraphicsManager->ReleaseTexture(texture_debug_view);
    }
}

static void TexturePreviewer(const TextureBase& texture, ImVec2 size = {64, 64}, ImVec2 uv0 = {0.0f, 0.0f}, ImVec2 uv1 = {1.0f, 1.0f}) {
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::Image((ImTextureID)texture.handler, size, uv0, uv1);
    if (ImGui::IsItemHovered())
    {
        float focus_sz = 32.0f;
        float focus_x = io.MousePos.x - pos.x - focus_sz * 0.5f;
        if (focus_x < 0.0f)
            focus_x = 0.0f;
        else if (focus_x > size.x - focus_sz)
            focus_x = size.x - focus_sz;
        float focus_y = io.MousePos.y - pos.y - focus_sz * 0.5f;
        if (focus_y < 0.0f)
            focus_y = 0.0f;
        else if (focus_y > size.y - focus_sz)
        focus_y = size.y - focus_sz;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(pos.x + focus_x, pos.y + focus_y), ImVec2(pos.x + focus_x + focus_sz, pos.y + focus_y + focus_sz), ImColor(255, 255, 0, 100));
        ImGui::BeginTooltip();
        ImVec2 _uv0 = ImVec2((focus_x) / size.x, (focus_y) / size.y);
        ImVec2 _uv1 = ImVec2((focus_x + focus_sz) / size.x, (focus_y + focus_sz) / size.y);
        ImGui::Text("UV(min): (%.2f, %.2f)", _uv0.x, _uv0.y);
        ImGui::Text("UV(max): (%.2f, %.2f)", _uv1.x, _uv1.y);
        if (uv0.y > uv1.y) { _uv0.y = 1.0f - _uv0.y; _uv1.y = 1.0f - _uv1.y; }
        ImGui::Image((ImTextureID)texture.handler, ImVec2(128, 128), _uv0, _uv1, ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
        ImGui::EndTooltip();
    }
}

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
	    static bool show_app_metrics = false;
	    static bool show_app_debug_panel = true;
	    static bool show_app_about = false;
        size_t texture_view_index = 0;

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        // viewer
        {
            auto viewer_window_size = ImVec2(800, 600);
            ImGui::SetNextWindowSize(viewer_window_size, ImGuiCond_FirstUseEver);
            ImGui::Begin("Viewer");
            auto rect = ImGui::GetContentRegionAvail();
            if (frame.enableMSAA) {
                //m_pGraphicsManager->MSAAResolve(frame.colorTextures[0], frame.colorTextures[1]);
            }
            TexturePreviewer(frame.colorTextures[1], rect, {0.0f, 1.0f}, {1.0f, 0.0f});
            m_pGraphicsManager->ResizeCanvas(rect.x, rect.y);
            ImGui::End();
        }

        if (show_app_metrics)
        {
            ImGui::ShowMetricsWindow(&show_app_metrics);
        }

        if (show_app_about)
        {
            ImGui::Begin((const char*)u8"关于《从零开始手敲次世代游戏引擎》系列", &show_app_about, ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("Game Engine From Scratch, v0.1.0");
            ImGui::Separator();
            ImGui::Text((const char*)u8"由陈文礼和所有社区贡献者创建");
            ImGui::Text((const char*)u8"本项目代码采用 MIT 许可。对应文章采用 CC-BY 许可。具体请参照各自的许可证说明。");
            ImGui::End();
        }

        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu((const char*)u8"调试"))
            {
			    ImGui::MenuItem((const char*)u8"调试窗口", NULL, &show_app_debug_panel);
			    ImGui::MenuItem((const char*)u8"ImGui状态及调试窗口", NULL, &show_app_metrics);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu((const char*)u8"帮助"))
            {
			    ImGui::MenuItem((const char*)u8"关于本应用", NULL, &show_app_about);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        if (show_app_debug_panel) {
            static std::deque<float> fps_data;

            ImGui::Begin((const char*)u8"调试窗口");  // Create a debug window

            ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
            if (ImGui::CollapsingHeader((const char*)u8"运行状态与基本参数")) {
                ImGui::ColorEdit3(
                    (const char*)u8"清屏色",
                    (float*)frame.clearColor);  // Edit 3 floats representing a color

                float fps = ImGui::GetIO().Framerate;
                ImGui::Text((const char*)u8"平均渲染时间 %.3f 毫秒/帧 (%.1f FPS)",
                            1000.0f / fps,
                            fps);
                
                fps_data.push_back(fps);
                if (fps_data.size() > 100) fps_data.pop_front();

                auto getData = [](void* data, int index)->float {
                    return ((decltype(fps_data)*)data)->at(index);
                };

                ImGui::PlotLines((const char*)u8"帧率", getData, (void *)&fps_data, fps_data.size(), 0, "FPS", 0.0f);
            }


            ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
            if (ImGui::CollapsingHeader((const char*)u8"全局贴图")) {
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
                auto start_index = texture_view_index;

                for (int32_t i = 0; i < 6; i++) {
                    Texture2D texture_debug_view;

                    if (texture_view_index >= m_TextureViews.size()) {
                        m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i, 0);
                        m_TextureViews.push_back(texture_debug_view);
                    } 

                    ++texture_view_index;
                }

                for (int32_t i = 0; i < 6; i++) {
                    Texture2D texture_debug_view;

                    if (texture_view_index >= m_TextureViews.size()) {
                        m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i, 1);
                        m_TextureViews.push_back(texture_debug_view);
                    } 

                    ++texture_view_index;
                }

                for (int32_t i = 6; i < 12; i++) {
                    Texture2D texture_debug_view;

                    if (texture_view_index >= m_TextureViews.size()) {
                        m_pGraphicsManager->CreateTextureView(texture_debug_view, frame.skybox, i, 0);
                        m_TextureViews.push_back(texture_debug_view);
                    } 

                    ++texture_view_index;
                }

                if (ImGui::BeginTabBar((const char*)u8"环境贴图")) {
                    constexpr float spacing = 4;
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));

                    if (ImGui::BeginTabItem((const char*)u8"天空盒")) {
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

                        ImGui::EndTabItem();
                    }

                    start_index += 6;

                    // Draw Irradiance
                    if (ImGui::BeginTabItem((const char*)u8"辐照度")) {


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

                        ImGui::EndTabItem();
                    }

                    start_index += 6;

                    // Draw Radiance
                    if (ImGui::BeginTabItem((const char*)u8"辐射度")) {
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

                        ImGui::EndTabItem();
                    }

                    ImGui::EndTabBar();

                    ImGui::PopStyleVar();
                }

                // BRDF LUT
                ImGui::Text(
                    (const char*)u8"BRDF LUT");

                TexturePreviewer(frame.brdfLUT);
            }

            ImGui::End(); // finish the debug window
        }
    }
}