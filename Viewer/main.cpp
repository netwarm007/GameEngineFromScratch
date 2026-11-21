#include "GfxConfiguration.hpp"
#include "config.h"

#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "GuiContext.hpp"
#include "OGEX.hpp"
#include "Bullet/BulletPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "SceneResourceLoader.hpp"
#include "ViewerLogic.hpp"
#include "SplashSequence.hpp"

#if defined(MYGE_RUNTIME_RHI)
#include "RhiComposition.hpp"

#include <iostream>
#else
#if defined(MYGE_USE_VULKAN)
#include "VulkanApplication.hpp"
#include "RHI/Vulkan/VulkanGraphicsManager.hpp"
#elif defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#elif defined(OS_WINDOWS)
#if defined(VIEWER_USE_WIN32_OPENGL)
#include "Platform/Windows/OpenGLApplication.hpp"
#else
#include "D3d12Application.hpp"
#endif
#else
#if defined(HAS_SDL2)
#include "Platform/Sdl/OpenGLApplication.hpp"
#else
#include "Platform/Linux/OpenGLApplication.hpp"
#endif
#endif

#if defined(MYGE_USE_VULKAN)
#elif defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#elif defined(OS_WINDOWS) && !defined(VIEWER_USE_WIN32_OPENGL)
#include "RHI/D3d/D3d12Config.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif
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
    GuiContext gui;

#if defined(MYGE_RUNTIME_RHI)
    auto parsed = ParseRhiStartupArguments(
        std::span<char* const>(argv, static_cast<size_t>(argc)));
    if (!parsed.options) {
        std::cerr << parsed.error << '\n';
        return 1;
    }
    RhiStartupOptions& rhiOptions = *parsed.options;
    const auto availability = ProbeRhiStartupAvailability();
    if (rhiOptions.listBackends) {
        for (const auto& entry : availability) {
            std::cout << RhiBackendName(entry.backend) << ": ";
            switch (entry.state) {
                case RhiAvailability::NotCompiled:
                    std::cout << "not compiled";
                    break;
                case RhiAvailability::RuntimeUnavailable:
                    std::cout << "runtime unavailable";
                    break;
                case RhiAvailability::ReadyForInitialization:
                    std::cout << "ready to attempt initialization";
                    break;
            }
            if (!entry.reason.empty()) std::cout << " (" << entry.reason << ')';
            std::cout << '\n';
        }
        return 0;
    }
#if defined(OS_WINDOWS)
    constexpr RhiBackend defaultBackend = RhiBackend::D3d12;
#elif defined(OS_MACOS)
    constexpr RhiBackend defaultBackend = RhiBackend::Metal;
#else
    constexpr RhiBackend defaultBackend = RhiBackend::OpenGL;
#endif
    const auto selection = ResolveRhiStartupBackend(
        rhiOptions.requestedBackend, defaultBackend, availability);
    if (!selection.backend) {
        std::cerr << selection.error << '\n';
        return 1;
    }
    const std::string windowTitle = std::string("Viewer - ") +
        std::string(RhiStartupPlatformName()) + "/" +
        std::string(RhiBackendName(*selection.backend)) + " (" +
        std::string(RhiStartupHostName(*selection.backend)) + ")";
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, windowTitle.c_str());
#else
#if defined(VIEWER_USE_WIN32_OPENGL)
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, "Viewer (Win32/OpenGL)");
#else
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, "Viewer");
#endif
#if defined(IS_OPENGL)
    config.fixOpenGLPerspectiveMatrix = true;
#endif
#endif
#if !defined(MYGE_RUNTIME_RHI)
#if defined(MYGE_USE_VULKAN)
    VulkanApplication app(config);
#elif defined(OS_MACOS)
    CocoaMetalApplication app(config);
#elif defined(OS_WINDOWS) && !defined(VIEWER_USE_WIN32_OPENGL)
    D3d12Application app(config);
#else
    OpenGLApplication app(config);
#endif
#endif
    AssetLoader assetLoader;
    MemoryManager memoryManager;
    OgexParser sceneParser;
    SceneResourceLoader sceneResources(assetLoader);
    SceneManager sceneManager(assetLoader, sceneParser, sceneResources);
    ViewerLogic gameLogic(sceneManager);
    BulletPhysicsManager physicsManager(sceneManager);
    InputManager inputManager(gameLogic);
    AnimationManager animationManager(sceneManager);
    SplashSequence splashSequence(sceneManager, physicsManager, animationManager);
#ifdef DEBUG
    DebugManager debugManager;
#endif
#if defined(MYGE_RUNTIME_RHI)
    auto composed = CreateRhiStartupComposition(
        *selection.backend, config, sceneManager, assetLoader);
    if (!composed.composition) {
        std::cerr << composed.error << '\n';
        return 1;
    }
    BaseApplication& app = *composed.composition->application;
    IGraphicsManager* graphicsManager = composed.composition->graphicsManager.get();
    IPipelineStateManager* pipelineStateManager =
        composed.composition->pipelineStateManager.get();
#elif defined(MYGE_USE_VULKAN)
    VulkanPipelineStateManager pipelineStateManager(app.GetRHI(), assetLoader);
    VulkanGraphicsManager graphicsManager(app.GetRHI(), sceneManager, pipelineStateManager);
#else
    TPipelineStateManager pipelineStateManager;
    TGraphicsManager graphicsManager(sceneManager, pipelineStateManager);
#endif

#if defined(MYGE_RUNTIME_RHI)
    // The borrowed argument strings and filtered pointer array live in
    // rhiOptions until main exits, outliving the application.
    app.SetCommandLineParameters(
        static_cast<int>(rhiOptions.applicationArguments.size()) - 1,
        rhiOptions.applicationArguments.data());
#else
    app.SetCommandLineParameters(argc, argv);
#endif

    app.RegisterManagerModule(&animationManager);
    app.RegisterManagerModule(&splashSequence);
    app.RegisterManagerModule(&assetLoader);
#if defined(MYGE_RUNTIME_RHI)
    app.RegisterManagerModule(graphicsManager);
#else
    app.RegisterManagerModule(&graphicsManager);
#endif
    app.RegisterManagerModule(&inputManager);
    app.RegisterManagerModule(&memoryManager);
    app.RegisterManagerModule(&physicsManager);
#if defined(MYGE_RUNTIME_RHI)
    app.RegisterManagerModule(pipelineStateManager);
#else
    app.RegisterManagerModule(&pipelineStateManager);
#endif
    app.RegisterManagerModule(&sceneManager);
    app.RegisterManagerModule(&gameLogic);
#ifdef DEBUG
    app.RegisterManagerModule(&debugManager);
#endif

    // Initialize ImGui
    ret = gui.Initialize();
    if (ret != 0) return ret;
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    // The executable path must be known before resolving asset paths.
    ret = assetLoader.Initialize();
    if (ret != 0) return ret;
    auto font_path = assetLoader.GetFileRealPath("Fonts/NotoSansMonoCJKsc-VF.ttf");

    ImVector<ImWchar> ranges;
    ImFontGlyphRangesBuilder builder;
    builder.AddText((const char*)u8"辐屏擎渲帧钮");                        // Add a string (here "Hello world" contains 7 unique characters)
    builder.AddRanges(io.Fonts->GetGlyphRangesChineseSimplifiedCommon()); // Add one of the default ranges
    builder.BuildRanges(&ranges);                          // Build the final result (ordered ranges with all the unique characters submitted)

    io.Fonts->AddFontFromFileTTF(font_path.c_str(), 16.0f, NULL, ranges.Data);

    ImGui::StyleColorsDark();

    ImGuiStyle& im_style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        im_style.WindowRounding = 0.0f;
        im_style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Create Main Window
    app.CreateMainWindow();

    ret = app.Initialize();
    if (ret != 0) {
        app.Finalize();
        return ret;
    }

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

    return ret;
}
