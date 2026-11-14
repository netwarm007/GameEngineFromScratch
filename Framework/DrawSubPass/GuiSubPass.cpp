#include "GuiSubPass.hpp"
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"

#include <cmath>
#include <deque>

#include "AssetLoader.hpp"
#include "PNG.hpp"

using namespace My;

GuiSubPass::~GuiSubPass() {
    for (auto& texture_debug_view : m_TextureViews) {
        m_pGraphicsManager->ReleaseTexture(texture_debug_view);
    }

    for (auto& icon : m_SceneIcons) {
        if (icon.handler) m_pGraphicsManager->ReleaseTexture(icon);
    }
}

namespace {
constexpr std::array<const char*, 5> kSceneIconFiles{
    "Textures/Icons/scene_empty.png", "Textures/Icons/scene_geometry.png",
    "Textures/Icons/scene_light.png", "Textures/Icons/scene_camera.png",
    "Textures/Icons/scene_bone.png"};

const char* SceneNodeKindName(SceneNodeKind kind) {
    switch (kind) {
        case SceneNodeKind::Geometry:
            return (const char*)u8"几何体";
        case SceneNodeKind::Light:
            return (const char*)u8"灯光";
        case SceneNodeKind::Camera:
            return (const char*)u8"相机";
        case SceneNodeKind::Bone:
            return (const char*)u8"骨骼";
        default:
            return (const char*)u8"空节点";
    }
}
}  // namespace

void GuiSubPass::LoadSceneIcons() {
    m_SceneIconsLoaded = true;

    AssetLoader loader;
    PngParser parser;
    for (size_t i = 0; i < kSceneIconFiles.size(); i++) {
        if (!loader.FileExists(kSceneIconFiles[i])) {
            std::cerr << "[GuiSubPass] Icon not found: " << kSceneIconFiles[i]
                      << std::endl;
            continue;
        }
        Buffer buf = loader.SyncOpenAndReadBinary(kSceneIconFiles[i]);
        Image image = parser.Parse(buf);
        m_SceneIcons[i] = m_pGraphicsManager->CreateTexture(image);
    }
}

void GuiSubPass::DrawSceneNode(const std::shared_ptr<BaseSceneNode>& node) {
    const auto children = node->GetChildNodes();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                               ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_DefaultOpen;
    if (children.empty()) flags |= ImGuiTreeNodeFlags_Leaf;
    if (m_SelectedNode.lock() == node) flags |= ImGuiTreeNodeFlags_Selected;

    const auto kind = node->GetKind();
    const auto& icon = m_SceneIcons[static_cast<size_t>(kind)];
    const float icon_size = ImGui::GetTextLineHeight();

    ImGui::PushID(node.get());
    const bool open = ImGui::TreeNodeEx("##node", flags);
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        m_SelectedNode = node;
    }
    ImGui::SameLine();
    if (icon.handler) {
        ImGui::Image((ImTextureID)icon.handler, ImVec2(icon_size, icon_size));
    } else {
        ImGui::Text("[%s]", SceneNodeKindName(kind));
    }
    ImGui::SameLine();
    const auto name = node->GetName();
    ImGui::TextUnformatted(name.empty() ? (const char*)u8"(未命名)"
                                        : name.c_str());
    if (open) {
        for (const auto& child : children) DrawSceneNode(child);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void GuiSubPass::DrawSceneNodeDetail(const BaseSceneNode& node) {
    const auto kind = node.GetKind();
    ImGui::Text((const char*)u8"名称: %s", node.GetName().c_str());
    ImGui::Text((const char*)u8"类型: %s", SceneNodeKindName(kind));
    ImGui::Text((const char*)u8"子节点数: %zu", node.GetChildCount());
    ImGui::Text((const char*)u8"动画剪辑数: %zu", node.GetAnimationClipCount());

    switch (kind) {
        case SceneNodeKind::Geometry: {
            const auto& n = static_cast<const SceneGeometryNode&>(node);
            ImGui::Text((const char*)u8"对象引用: %s",
                        n.GetSceneObjectRef().c_str());
            ImGui::Text((const char*)u8"可见: %s  投射阴影: %s  运动模糊: %s",
                        n.Visible() ? "Y" : "N", n.CastShadow() ? "Y" : "N",
                        n.MotionBlur() ? "Y" : "N");
            break;
        }
        case SceneNodeKind::Light: {
            const auto& n = static_cast<const SceneLightNode&>(node);
            ImGui::Text((const char*)u8"对象引用: %s",
                        n.GetSceneObjectRef().c_str());
            ImGui::Text((const char*)u8"投射阴影: %s",
                        n.CastShadow() ? "Y" : "N");
            break;
        }
        case SceneNodeKind::Camera: {
            const auto& n = static_cast<const SceneCameraNode&>(node);
            ImGui::Text((const char*)u8"对象引用: %s",
                        n.GetSceneObjectRef().c_str());
            const auto& t = n.GetTarget();
            ImGui::Text((const char*)u8"目标: (%.3f, %.3f, %.3f)", t[0], t[1],
                        t[2]);
            break;
        }
        default:
            break;
    }

    ImGui::Separator();
    const auto m = *node.GetCalculatedTransform();
    ImGui::Text((const char*)u8"局部变换");
    ImGui::Text((const char*)u8"  平移: (%.3f, %.3f, %.3f)", m[3][0], m[3][1],
                m[3][2]);
    auto row_len = [&m](int r) {
        return std::sqrt(m[r][0] * m[r][0] + m[r][1] * m[r][1] +
                         m[r][2] * m[r][2]);
    };
    ImGui::Text((const char*)u8"  缩放: (%.3f, %.3f, %.3f)", row_len(0),
                row_len(1), row_len(2));
}

void GuiSubPass::DrawSceneGraphWindow(const Frame& frame, bool* open) {
    if (!m_SceneIconsLoaded) LoadSceneIcons();

    if (!ImGui::Begin((const char*)u8"场景图", open)) {
        ImGui::End();
        return;
    }

    const auto scene = frame.scene;
    if (scene.get() != m_LastScene) {
        m_LastScene = scene.get();
        m_SelectedNode.reset();
    }

    if (!scene || !scene->SceneGraph) {
        ImGui::TextUnformatted((const char*)u8"无场景");
        ImGui::End();
        return;
    }

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const bool stack_details = available.x < 420.0f;
    ImGui::BeginChild("scene_tree",
                     ImVec2(stack_details ? 0.0f : available.x * 0.5f,
                            stack_details ? available.y * 0.6f : 0.0f),
                     true);
    DrawSceneNode(scene->SceneGraph);
    ImGui::EndChild();

    if (!stack_details) ImGui::SameLine();
    ImGui::BeginChild("scene_node_detail", ImVec2(0, 0), true);
    if (auto selected = m_SelectedNode.lock()) {
        DrawSceneNodeDetail(*selected);
    } else {
        ImGui::TextUnformatted((const char*)u8"请选择一个节点");
    }
    ImGui::EndChild();

    ImGui::End();
}

static void TexturePreviewer(const TextureBase& texture, ImVec2 size = {64, 64}, ImVec2 uv0 = {0.0f, 1.0f}, ImVec2 uv1 = {1.0f, 0.0f}) {
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
        const std::array<TextureHandler, 4> sources{
            frame.frameContext.globalShadowMap.handler, frame.frameContext.shadowMap.handler,
            frame.frameContext.cubeShadowMap.handler, frame.skybox.handler};
        const std::array<uint32_t, 3> sizes{
            frame.frameContext.globalShadowMap.size, frame.frameContext.shadowMap.size,
            frame.frameContext.cubeShadowMap.size};
        if (m_viewSources != sources || m_viewSizes != sizes) {
            for (auto& view : m_TextureViews) m_pGraphicsManager->ReleaseTexture(view);
            m_TextureViews.clear();
            m_viewSources = sources;
            m_viewSizes = sizes;
        }
	    static bool show_app_metrics = false;
	    static bool show_app_debug_panel = true;
	    static bool show_scene_graph = true;
	    static bool show_app_about = false;
        size_t texture_view_index = 0;

        const auto* viewport = ImGui::GetMainViewport();
        const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, viewport);
        if (!m_DockLayoutInitialized || m_ResetDockLayout) {
            const auto* node = ImGui::DockBuilderGetNode(dockspace);
            const bool has_saved_layout = ImGui::FindWindowSettingsByID(ImHashStr("Viewer")) != nullptr;
            if (m_ResetDockLayout ||
                (!has_saved_layout && node && !node->IsSplitNode() && node->Windows.empty())) {
                ImGui::DockBuilderRemoveNode(dockspace);
                ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodePos(dockspace, viewport->WorkPos);
                ImGui::DockBuilderSetNodeSize(dockspace, viewport->WorkSize);
                ImGuiID viewer_dock = dockspace;
                ImGuiID debug_dock;
                ImGuiID scene_dock;
                ImGui::DockBuilderSplitNode(viewer_dock, ImGuiDir_Down, 0.28f,
                                            &debug_dock, &viewer_dock);
                ImGui::DockBuilderSplitNode(viewer_dock, ImGuiDir_Left, 0.28f,
                                            &scene_dock, &viewer_dock);
                ImGui::DockBuilderDockWindow("Viewer", viewer_dock);
                ImGui::DockBuilderDockWindow((const char*)u8"场景图", scene_dock);
                ImGui::DockBuilderDockWindow((const char*)u8"调试窗口", debug_dock);
                ImGui::DockBuilderDockWindow("Dear ImGui Metrics/Debugger", debug_dock);
                ImGui::DockBuilderFinish(dockspace);
            }
            m_DockLayoutInitialized = true;
            m_ResetDockLayout = false;
        }

        // viewer
        {
            ImGui::Begin("Viewer");
            auto rect = ImGui::GetContentRegionAvail();
            if (frame.enableMSAA) {
                m_pGraphicsManager->MSAAResolve(frame.colorTextures[0], frame.colorTextures[1]);
            }
            if (m_pGraphicsManager->RenderTargetHasTopLeftOrigin()) {
                TexturePreviewer(frame.colorTextures[0], rect, {0.0f, 0.0f}, {1.0f, 1.0f});
            } else {
                TexturePreviewer(frame.colorTextures[0], rect);
            }
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

        if (show_scene_graph) {
            DrawSceneGraphWindow(frame, &show_scene_graph);
        }

        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu((const char*)u8"调试"))
            {
			    ImGui::MenuItem((const char*)u8"调试窗口", NULL, &show_app_debug_panel);
			    ImGui::MenuItem((const char*)u8"场景图", NULL, &show_scene_graph);
			    ImGui::MenuItem((const char*)u8"ImGui状态及调试窗口", NULL, &show_app_metrics);
                if (ImGui::MenuItem((const char*)u8"重置窗口布局")) {
                    m_ResetDockLayout = true;
                    show_scene_graph = show_app_debug_panel = true;
                }
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

                if (frame.skybox.handler) {
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
                        // -Y
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(1);
                        TexturePreviewer(m_TextureViews[start_index + 3]);
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
                        // +Y
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(1);
                        TexturePreviewer(m_TextureViews[start_index + 2]);

                        ImGui::EndTable();

                        ImGui::EndTabItem();
                    }

                    start_index += 6;

                    // Draw Irradiance
                    if (ImGui::BeginTabItem((const char*)u8"辐照度")) {


                        ImGui::BeginTable("irradiance", 4);
                        // -Y
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(1);
                        TexturePreviewer(m_TextureViews[start_index + 3]);
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
                        // +Y
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(1);
                        TexturePreviewer(m_TextureViews[start_index + 2]);

                        ImGui::EndTable();

                        ImGui::EndTabItem();
                    }

                    start_index += 6;

                    // Draw Radiance
                    if (ImGui::BeginTabItem((const char*)u8"辐射度")) {
                        ImGui::BeginTable("radiance", 4);
                        // -Y
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(1);
                        TexturePreviewer(m_TextureViews[start_index + 3]);
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
                        // +Y
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(1);
                        TexturePreviewer(m_TextureViews[start_index + 2]);

                        ImGui::EndTable();

                        ImGui::EndTabItem();
                    }

                    ImGui::EndTabBar();

                    ImGui::PopStyleVar();
                }

                // BRDF LUT
                }
                ImGui::Text(
                    (const char*)u8"BRDF LUT");

                TexturePreviewer(frame.brdfLUT);
            }

            ImGui::End(); // finish the debug window
        }
    }
}