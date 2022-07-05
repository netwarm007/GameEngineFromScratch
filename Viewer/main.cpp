#include "GfxConfiguration.hpp"
#include "config.h"

#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "My/MyPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "ViewerLogic.hpp"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#elif defined(OS_WINDOWS)
//#include "D3d12Application.hpp"
#include "OpenGLApplication.hpp"
#else
#if defined(HAS_SDL2)
#include "Platform/Sdl/OpenGLApplication.hpp"
#else
#include "Platform/Linux/OpenGLApplication.hpp"
#endif
#endif

#if defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#elif defined(OS_WINDOWS)
//#include "RHI/D3d/D3d12Config.hpp"
#include "RHI/OpenGL/OpenGLConfig.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif

#if defined(OS_WEBASSEMBLY)
#include <emscripten.h>

#include <functional>

std::function<void()> loop;
void main_loop() { loop(); }
#endif  // defined(OS_WEBASSEMBLY)

#include "imgui.h"

using namespace My;

int main(int argc, char** argv) {
    int ret;

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, "Viewer");
#if defined(IS_OPENGL)
    config.fixOpenGLPerspectiveMatrix = true;
#endif
#if defined(OS_MACOS)
    CocoaMetalApplication app(config);
#elif defined(OS_WINDOWS)
    //D3d12Application app(config);
    OpenGLApplication app(config);
#else
    OpenGLApplication app(config);
#endif
    ViewerLogic gameLogic;
    MyPhysicsManager physicsManager;
    AssetLoader assetLoader;
    MemoryManager memoryManager;
    SceneManager sceneManager;
    InputManager inputManager;
    AnimationManager animationManager;
#ifdef DEBUG
    DebugManager debugManager;
#endif
    TGraphicsManager graphicsManager;
    TPipelineStateManager pipelineStateManager;

    app.SetCommandLineParameters(argc, argv);

    app.RegisterManagerModule(&animationManager);
    app.RegisterManagerModule(&assetLoader);
    app.RegisterManagerModule(&graphicsManager);
    app.RegisterManagerModule(&inputManager);
    app.RegisterManagerModule(&memoryManager);
    app.RegisterManagerModule(&physicsManager);
    app.RegisterManagerModule(&pipelineStateManager);
    app.RegisterManagerModule(&sceneManager);
    app.RegisterManagerModule(&gameLogic);
#ifdef DEBUG
    app.RegisterManagerModule(&debugManager);
#endif

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    auto font_path = assetLoader.GetFileRealPath("Fonts/NotoSansMonoCJKsc-VF.ttf");

    ImVector<ImWchar> ranges;
    ImFontGlyphRangesBuilder builder;
    builder.AddText((const char*)u8"辐屏擎渲帧钮");                        // Add a string (here "Hello world" contains 7 unique characters)
    builder.AddRanges(io.Fonts->GetGlyphRangesChineseSimplifiedCommon()); // Add one of the default ranges
    builder.BuildRanges(&ranges);                          // Build the final result (ordered ranges with all the unique characters submitted)

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

    ret = app.Initialize();

    // Drive the modules ahead
#if defined(OS_WEBASSEMBLY)
    auto main_loop = [] { app.Tick(); };

    emscripten_set_main_loop(main_loop, 0, true);
#else
    while (!app.IsQuit()) {
        app.Tick();
    }
#endif

#if !defined(OS_WEBASSEMBLY)
    // Finalize App
    app.Finalize();
#endif

    ImGui::DestroyContext();

    return ret;
}
