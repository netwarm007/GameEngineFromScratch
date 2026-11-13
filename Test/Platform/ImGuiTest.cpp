#include "Platform/CrossPlatformGfxApp.hpp"
#include "Platform/TGraphicsManager.hpp"

#include "AssetLoader.hpp"

#include "imgui/imgui.h"

#include "RenderGraph/RenderPipeline/RenderPipeline.hpp"

using namespace My;

RenderGraph::RenderPipeline renderPipeline;

int test(BaseApplication& app) {
    int result;

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    AssetLoader assetLoader;
    auto font_path =
        assetLoader.GetFileRealPath("Fonts/NotoSansMonoCJKsc-VF.ttf");

    ImVector<ImWchar> ranges;
    ImFontGlyphRangesBuilder builder;
    builder.AddText(
        (const char*)u8"屏擎渲帧钮");

    builder.AddRanges(
        io.Fonts->GetGlyphRangesChineseSimplifiedCommon());  // Add one of the
                                                             // default ranges
    builder.BuildRanges(&ranges);  // Build the final result (ordered ranges
                                   // with all the unique characters submitted)

    io.Fonts->AddFontFromFileTTF(font_path.c_str(), 16.0f, NULL, ranges.Data);
    io.Fonts->Build();

    ImGui::StyleColorsDark();

    ImGuiStyle& im_style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        im_style.WindowRounding = 0.0f;
        im_style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Create Main Window
    app.CreateMainWindow();

    result = app.Initialize();

    if (result == 0) {
        while (!app.IsQuit()) {
            app.Tick();
        }

        app.Finalize();
    }

    ImGui::DestroyContext();

    return result;
}

class TestGraphicsManager : public TGraphicsManager {
    void Draw() override {
        static bool show_demo_window = true;
        auto& frame = m_Frames[m_nFrameIndex];
        BeginPass(frame);

        if (show_demo_window) {
            ImGui::ShowDemoWindow(&show_demo_window);
        }
        // 2. Show a simple window that we create ourselves. We use a Begin/End
        // pair to created a named window.
        {
            static float f1 = 0.0f;
            static float f2 = 0.0f;
            static int counter = 0;

            ImGui::Begin(
                (const char*)u8"你好，引擎！");  // Create a window called
                                                 // "Hello, world!" and append
                                                 // into it.

            ImGui::Text(
                (const char*)u8"这里有一些重要信息。");  // Display some text
                                                         // (you can use a
                                                         // format strings too)
            ImGui::Checkbox((const char*)u8"演示窗口",
                            &show_demo_window);  // Edit bools storing our
                                                 // window open/close state

            ImGui::SliderFloat(
                (const char*)u8"浮点数##1", &f1, 0.0f,
                1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::SliderFloat(
                (const char*)u8"浮点数##2", &f2, 0.0f,
                1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3(
                (const char*)u8"清屏色",
                (float*)
                    frame.clearColor);  // Edit 3 floats representing a color

            if (ImGui::Button(
                    (const char*)u8"按钮"))  // Buttons return true when clicked
                                             // (most widgets return true when
                                             // edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text((const char*)u8"按压次数 = %d", counter);

            ImGui::Text((const char*)u8"平均渲染时间 %.3f 毫秒/帧 (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();

            renderPipeline.reflectUI();
        }
        EndPass(frame);
    }
};

int main(int argc, char** argv) {
    int result = 0;

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "ImGUI Test");

    auto pApp = My::CreateApplication(config);

    TestGraphicsManager graphicsManager;

    pApp->SetCommandLineParameters(argc, argv);
    pApp->RegisterManagerModule(&graphicsManager);

    renderPipeline.render_passes.resize(3);

    result |= test(*pApp);

    delete pApp;

    return result;
}