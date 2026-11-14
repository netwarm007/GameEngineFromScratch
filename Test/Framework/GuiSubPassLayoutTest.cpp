#include <cstdlib>
#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "GraphicsManager.hpp"
#include "GuiSubPass.hpp"
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
    void ResizeCanvas(int32_t, int32_t) override {}
};

void TestLayout(ImVec2 position, ImVec2 displaySize, float framebufferScale) {
    ImGui::CreateContext();
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
        GuiSubPass gui(&graphics, nullptr);
        Frame frame;
        frame.colorTextures.resize(1);

        for (int i = 0; i < 3; i++) {
            ImGui::NewFrame();
            gui.Draw(frame);
            ImGui::Render();
            ImGui::UpdatePlatformWindows();
        }

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
            GuiSubPass restoredGui(&graphics, nullptr);
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
    ImGui::DestroyContext();
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _CALL_REPORTFAULT);
#endif
    AssetLoader loader;
    if (loader.Initialize() != 0) {
        std::cerr << "FAILED: AssetLoader initialization" << std::endl;
        return 1;
    }
    TestLayout(ImVec2(0, 0), ImVec2(1024, 768), 1.0f);
    TestLayout(ImVec2(640, 240), ImVec2(1024, 768), 2.0f);
    TestLayout(ImVec2(150, 100), ImVec2(1920, 1080), 1.0f);
    loader.Finalize();
    return g_failures;
}
