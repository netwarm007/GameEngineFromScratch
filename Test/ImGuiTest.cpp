#include "GfxConfiguration.hpp"
#include "config.h"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#elif defined(OS_WINDOWS)
#include "D3d12Application.hpp"
#else
#include "OpenGLApplication.hpp"
#endif

#if defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#elif defined(OS_WINDOWS)
#include "RHI/D3d/D3d12Config.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif

#include "AssetLoader.hpp"

#include "imgui/imgui.h"

using namespace My;

int test(BaseApplication& app) {
    int result;

    app.CreateMainWindow();

    result = app.Initialize();

    AssetLoader assetLoader;
    auto font_path = assetLoader.GetFileRealPath("Fonts/NotoSansCJKsc-VF.ttf");

    ImGuiIO& io = ImGui::GetIO();
    ImVector<ImWchar> ranges;
    ImFontGlyphRangesBuilder builder;
    builder.AddText((const char*)u8"屏擎渲帧钮");                        // Add a string (here "Hello world" contains 7 unique characters)
    builder.AddRanges(io.Fonts->GetGlyphRangesChineseSimplifiedCommon()); // Add one of the default ranges
    builder.BuildRanges(&ranges);                          // Build the final result (ordered ranges with all the unique characters submitted)

    io.Fonts->AddFontFromFileTTF(font_path.c_str(), 13.0f, NULL, ranges.Data);
    io.Fonts->Build();

    if (result == 0) {
        while (!app.IsQuit()) {
            app.Tick();
        }

        app.Finalize();
    }

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
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin((const char*)u8"你好，引擎！");  // Create a window called "Hello,
                                            // world!" and append into it.

            ImGui::Text(
                (const char*)u8"这里有一些重要信息。");  // Display some text (you can use
                                               // a format strings too)
            ImGui::Checkbox((const char*)u8"演示窗口",
                            &show_demo_window);  // Edit bools storing our
                                                 // window open/close state

            ImGui::SliderFloat(
                (const char*)u8"浮点数", &f, 0.0f,
                1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3(
                (const char*)u8"清屏色",
                (float*)frame.clearColor);  // Edit 3 floats representing a color

            if (ImGui::Button(
                    (const char*)u8"按钮"))  // Buttons return true when clicked (most
                                // widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text((const char*)u8"按压次数 = %d", counter);

            ImGui::Text((const char*)u8"平均渲染时间 %.3f 毫秒/帧 (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();
        }
        EndPass(frame);
    }
};

int main(int argc, char** argv) {
    int result;

#if defined(OS_MACOS)
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "ImGUI Test (Cocoa Meta)");
    CocoaMetalApplication app(config);
#endif

#if defined(OS_WINDOWS)
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "ImGUI Test (DX12)");
    D3d12Application app(config);
#endif

#if defined(OS_LINUX)
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                            "ImGUI Test (OpenGL)");
    OpenGLApplication app(config);
#endif

    TestGraphicsManager graphicsManager;

    app.SetCommandLineParameters(argc, argv);
    app.RegisterManagerModule(&graphicsManager);

    result |= test(app);

    return result;
}