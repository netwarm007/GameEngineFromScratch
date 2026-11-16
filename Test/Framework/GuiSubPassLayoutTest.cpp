#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

#include "GraphicsManager.hpp"
#include "GuiSubPass.hpp"
#include "GuiContext.hpp"
#include "imgui/imgui_internal.h"

using namespace My;

namespace {
int g_failures = 0;
ImVec2 g_windowPos;

void Expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << std::endl;
        g_failures++;
    }
}

class LayoutGraphicsManager : public GraphicsManager {
   public:
    int viewsCreated = 0;
    int viewsReleased = 0;
    void CreateTextureView(Texture2D& view, const TextureArrayBase& source,
                           uint32_t, uint32_t) override {
        view.handler = ++viewsCreated;
        view.width = source.width;
        view.height = source.height;
    }
    void ReleaseTexture(TextureBase&) override { ++viewsReleased; }

   private:
    void ResizeCanvas(int32_t, int32_t) override {}
};

void TestIconOwnership() {
    GuiContext context;
    Expect(context.Initialize() == 0, "icon regression context initializes");
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1024, 768);
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.Fonts->AddFontDefault();
    io.Fonts->Build();
    io.Fonts->SetTexID(1);

    class IconGraphics : public GraphicsManager {
       public:
        int uploads = 0;
        int releases = 0;
        std::set<TextureHandler> live;
        Texture2D CreateTexture(Image& image) override {
            Expect(image.data && image.Width == 1 && image.Height == 1 &&
                       image.data[0] == 123,
                   "injected decoded pixels reach the GPU upload unchanged");
            Texture2D texture;
            if (++uploads != 2) {
                texture.handler = 100 + uploads;
                live.insert(texture.handler);
            }
            return texture;
        }
        void ReleaseTexture(TextureBase& texture) override {
            Expect(live.erase(texture.handler) == 1,
                   "only successfully created icon textures are released once");
            ++releases;
        }
        void ResizeCanvas(int32_t, int32_t) override {}
    } graphics;

    std::vector<std::string> paths;
    auto load = [&](const char* path) {
        paths.emplace_back(path);
        Image image;
        if (paths.size() == 2 || paths.size() == 4) return image;
        image.Width = image.Height = 1;
        image.bitcount = 32;
        image.bitdepth = 8;
        image.pixel_format = PIXEL_FORMAT::RGBA8;
        image.pitch = image.data_size = 4;
        image.data = new uint8_t[4]{123, 45, 67, 255};
        return image;
    };
    Frame frame;
    frame.colorTextures.resize(1);
    frame.scene = std::make_shared<Scene>("icons A");
    auto draw = [&](GuiSubPass& gui) {
        ImGui::NewFrame();
        gui.Draw(frame);
        ImGui::Render();
    };
    {
        GuiSubPass unused(&graphics, nullptr, load);
    }
    Expect(paths.empty() && graphics.uploads == 0 && graphics.releases == 0,
           "destroying an undrawn GUI does not load or release icons");
    {
        GuiSubPass gui(&graphics, nullptr, load);
        draw(gui);
        draw(gui);
        frame.scene = std::make_shared<Scene>("icons B");
        draw(gui);
        Expect(paths == std::vector<std::string>{
                   "Textures/Icons/scene_empty.png",
                   "Textures/Icons/scene_geometry.png",
                   "Textures/Icons/scene_light.png",
                   "Textures/Icons/scene_camera.png",
                   "Textures/Icons/scene_bone.png"},
               "all icon paths load once despite frame and scene changes");
        Expect(graphics.uploads == 3 && graphics.live.size() == 2 &&
                   graphics.releases == 0,
               "unavailable images skip upload; failed upload does not discard other icons");
    }
    Expect(graphics.live.empty() && graphics.releases == 2,
           "GUI teardown releases successful icons including after partial failure");
    {
        GuiSubPass gui(&graphics, nullptr, load);
        draw(gui);
    }
    Expect(paths.size() == 10 && graphics.uploads == 8 &&
               graphics.releases == 7 && graphics.live.empty(),
           "a new GUI lifecycle reloads its own icons and releases them once");
}

void TestLayout(ImVec2 position, ImVec2 displaySize, float framebufferScale) {
    GuiContext context;
    if (context.Initialize() != 0) {
        Expect(false, "layout context initialization");
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = displaySize;
    io.DisplayFramebufferScale = ImVec2(framebufferScale, framebufferScale);
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;
    io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports |
                       ImGuiBackendFlags_RendererHasViewports;
    io.Fonts->AddFontDefault();
    io.Fonts->Build();
    io.Fonts->SetTexID(1);

    g_windowPos = position;
    auto& platform = ImGui::GetPlatformIO();
    platform.Platform_CreateWindow = [](ImGuiViewport* v) { v->PlatformHandle = v; };
    platform.Platform_DestroyWindow = [](ImGuiViewport* v) { v->PlatformHandle = nullptr; };
    platform.Platform_GetWindowPos = [](ImGuiViewport* v) {
        return v == ImGui::GetMainViewport() ? g_windowPos : v->Pos;
    };
    platform.Platform_SetWindowPos = [](ImGuiViewport*, ImVec2) {};
    platform.Platform_GetWindowSize = [](ImGuiViewport* v) { return v->Size; };
    platform.Platform_SetWindowSize = [](ImGuiViewport*, ImVec2) {};
    platform.Platform_ShowWindow = [](ImGuiViewport*) {};
    platform.Platform_SetWindowTitle = [](ImGuiViewport*, const char*) {};
    ImGuiPlatformMonitor monitor;
    monitor.MainSize = monitor.WorkSize = ImVec2(3840, 2160);
    monitor.DpiScale = framebufferScale;
    platform.Monitors.push_back(monitor);
    ImGui::GetMainViewport()->PlatformHandle = ImGui::GetMainViewport();

    {
        LayoutGraphicsManager graphics;
        GuiSubPass gui(&graphics, nullptr, [](const char*) { return Image{}; });
        Frame frame;
        frame.colorTextures.resize(1);
        frame.scene = std::make_shared<Scene>("A");
        frame.frameContext.globalShadowMap.handler = 42;
        frame.frameContext.globalShadowMap.size = 1;
        frame.frameContext.globalShadowMap.width = 64;
        frame.frameContext.globalShadowMap.height = 64;

        for (int i = 0; i < 3; i++) {
            ImGui::NewFrame();
            gui.Draw(frame);
            ImGui::Render();
            ImGui::UpdatePlatformWindows();
        }
        Expect(graphics.viewsCreated == 1 && graphics.viewsReleased == 0,
               "unchanged scene reuses its debug texture view");
        frame.scene = std::make_shared<Scene>("B");
        ImGui::NewFrame();
        gui.Draw(frame);
        ImGui::Render();
        ImGui::UpdatePlatformWindows();
        Expect(graphics.viewsCreated == 2 && graphics.viewsReleased == 1,
               "scene change refreshes debug views even when handles are reused");

        auto* scene = ImGui::FindWindowByName((const char*)u8"场景图");
        auto* debug = ImGui::FindWindowByName((const char*)u8"调试窗口");
        auto* viewer = ImGui::FindWindowByName("Viewer");
        Expect(scene && debug && viewer, "all default panels are created");
        if (scene && debug && viewer) {
            const auto* viewport = ImGui::GetMainViewport();
            const ImRect work(viewport->WorkPos,
                              ImVec2(viewport->WorkPos.x + viewport->WorkSize.x,
                                     viewport->WorkPos.y + viewport->WorkSize.y));
            Expect(scene->Active && !scene->Hidden && !scene->Collapsed,
                   "scene graph is visible by default");
            Expect(scene->Viewport == viewport, "scene graph stays in the captured main viewport");
            Expect(scene->DockIsActive && debug->DockIsActive && viewer->DockIsActive,
                   "all default panels use docking");
            Expect(scene->DockId != debug->DockId && scene->DockId != viewer->DockId &&
                       debug->DockId != viewer->DockId,
                   "panels occupy separate docking regions");
            Expect(work.Contains(scene->Rect()) && work.Contains(debug->Rect()) &&
                       work.Contains(viewer->Rect()),
                   "all panels fit inside the main work area");
            Expect(!scene->Rect().Overlaps(debug->Rect()) &&
                       !scene->Rect().Overlaps(viewer->Rect()) &&
                       !viewer->Rect().Overlaps(debug->Rect()),
                   "no default panels overlap");
            Expect(scene->Rect().Max.x <= viewer->Rect().Min.x &&
                       scene->Rect().Max.y <= debug->Rect().Min.y &&
                       viewer->Rect().Max.y <= debug->Rect().Min.y,
                   "scene graph is left, Viewer is right, and debug panel is below");
            Expect(ImGui::GetDrawData()->TotalVtxCount > 0,
                   "main viewport contains rendered UI geometry");

            const ImVec2 oldViewerSize = viewer->Size;
            io.DisplaySize = ImVec2(displaySize.x * 0.8f, displaySize.y * 0.8f);
            for (int i = 0; i < 3; i++) {
                ImGui::NewFrame();
                gui.Draw(frame);
                ImGui::Render();
                ImGui::UpdatePlatformWindows();
            }
            Expect(viewer->Size.x < oldViewerSize.x && viewer->Size.y < oldViewerSize.y,
                   "docked panels resize with the main viewport");
            Expect(!scene->Rect().Overlaps(viewer->Rect()) &&
                       !scene->Rect().Overlaps(debug->Rect()) &&
                       !viewer->Rect().Overlaps(debug->Rect()),
                   "resizing preserves non-overlapping regions");

            const ImVec2 savedPos(viewport->WorkPos.x + 100, viewport->WorkPos.y + 100);
            const ImVec2 savedSize(300, 250);
            ImGui::SetWindowDock(scene, 0, ImGuiCond_Always);
            ImGui::SetWindowPos(scene->Name, savedPos);
            ImGui::SetWindowSize(scene->Name, savedSize);
            const std::string settings = ImGui::SaveIniSettingsToMemory();
            ImGui::LoadIniSettingsFromMemory(settings.c_str());
            ImGui::NewFrame();
            GuiSubPass restoredGui(&graphics, nullptr,
                                  [](const char*) { return Image{}; });
            restoredGui.Draw(frame);
            ImGui::Render();
            ImGui::UpdatePlatformWindows();
            Expect(scene->Pos.x == savedPos.x && scene->Pos.y == savedPos.y &&
                       scene->Size.x == savedSize.x && scene->Size.y == savedSize.y &&
                       !scene->DockIsActive,
                   "restoring existing settings preserves custom floating layout");
        }
    }
    ImGui::DestroyPlatformWindows();
}

void TestContextOwnership() {
    GuiContext owner;
    Expect(owner.Initialize() == 0, "GUI context owner initializes");
    auto* context = ImGui::GetCurrentContext();
    auto* font = ImGui::GetIO().Fonts->AddFontDefault();
    Expect(owner.Initialize() != 0, "double context initialization is rejected");
    {
        GuiContext other;
        Expect(other.Initialize() != 0, "a live external context is not replaced");
    }
    Expect(ImGui::GetCurrentContext() == context &&
               ImGui::GetIO().Fonts->Fonts[0] == font,
           "failed initialization preserves the active context and its fonts");
    owner.Finalize();
    owner.Finalize();
    Expect(!ImGui::GetCurrentContext(), "context cleanup is repeatable");
    Expect(owner.Initialize() == 0, "context ownership supports a new lifecycle");
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _CALL_REPORTFAULT);
#endif
    TestLayout(ImVec2(0, 0), ImVec2(1024, 768), 1.0f);
    TestLayout(ImVec2(640, 240), ImVec2(1024, 768), 2.0f);
    TestLayout(ImVec2(150, 100), ImVec2(1920, 1080), 1.0f);
    TestIconOwnership();
    TestContextOwnership();
    return g_failures;
}
