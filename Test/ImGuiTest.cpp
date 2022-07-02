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

#include "imgui/imgui.h"

using namespace My;

int test(BaseApplication& app) {
    int result;

    app.CreateMainWindow();

    result = app.Initialize();

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

            ImGui::Begin("Hello, world!");  // Create a window called "Hello,
                                            // world!" and append into it.

            ImGui::Text(
                "This is some useful text.");  // Display some text (you can use
                                               // a format strings too)
            ImGui::Checkbox("Demo Window",
                            &show_demo_window);  // Edit bools storing our
                                                 // window open/close state

            ImGui::SliderFloat(
                "float", &f, 0.0f,
                1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3(
                "clear color",
                (float*)frame.clearColor);  // Edit 3 floats representing a color

            if (ImGui::Button(
                    "Button"))  // Buttons return true when clicked (most
                                // widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
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