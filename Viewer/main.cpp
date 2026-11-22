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
#include "Framework/Manager/RuntimeRoot.hpp"

#include <iostream>
#if defined(OS_WINDOWS)
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include <zlib.h>

#include "glad/glad.h"
#endif
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

#if defined(OS_WINDOWS) && defined(MYGE_RUNTIME_RHI)
namespace {
// Minimal 8-bit RGB PNG writer (filter 0), mirroring the D3D12/Vulkan
// self-capture contract without pulling in an image library.
bool WriteViewerCapturePng(const std::filesystem::path& path, uint32_t width,
                           uint32_t height,
                           const std::vector<unsigned char>& rgb) {
    auto put32 = [](std::vector<unsigned char>& out, uint32_t value) {
        for (int i = 3; i >= 0; --i) out.push_back((value >> (8 * i)) & 0xFF);
    };
    auto chunk = [&](std::vector<unsigned char>& out, const char* type,
                     const std::vector<unsigned char>& data) {
        put32(out, static_cast<uint32_t>(data.size()));
        const size_t start = out.size();
        out.insert(out.end(), type, type + 4);
        out.insert(out.end(), data.begin(), data.end());
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = start; i < out.size(); ++i) {
            crc ^= out[i];
            for (int k = 0; k < 8; ++k)
                crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
        put32(out, crc ^ 0xFFFFFFFFu);
    };
    std::vector<unsigned char> ihdr;
    put32(ihdr, width);
    put32(ihdr, height);
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // color type: truecolor RGB
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    const size_t stride = size_t(width) * 3;
    std::vector<unsigned char> raw;
    raw.reserve((stride + 1) * height);
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);  // filter: none
        // glReadPixels returns bottom-up rows; PNG stores top-down.
        const unsigned char* row = rgb.data() + size_t(height - 1 - y) * stride;
        raw.insert(raw.end(), row, row + stride);
    }
    uLongf bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<unsigned char> idat(bound);
    if (compress2(idat.data(), &bound, raw.data(),
                  static_cast<uLong>(raw.size()), Z_BEST_SPEED) != Z_OK) {
        return false;
    }
    idat.resize(bound);
    std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", idat);
    chunk(png, "IEND", {});
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(png.data()),
                 static_cast<std::streamsize>(png.size()));
    output.close();
    return static_cast<bool>(output);
}

bool CaptureViewerFrame(IApplication& app, const std::filesystem::path& path) {
    uint32_t width = 0, height = 0;
    app.GetFramebufferSize(width, height);
    if (!width || !height) {
        std::cerr << "Invalid Viewer capture dimensions\n";
        return false;
    }
    std::vector<unsigned char> pixels(size_t(width) * height * 3);
    GLint framebuffer, readBuffer, alignment;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_READ_BUFFER, &readBuffer);
    glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    const auto error = glGetError();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glReadBuffer(readBuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, alignment);
    if (error != GL_NO_ERROR) {
        std::cerr << "Viewer framebuffer capture failed: " << error << '\n';
        return false;
    }
    return WriteViewerCapturePng(path, width, height, pixels);
}
}  // namespace
#endif

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
#if defined(OS_WINDOWS)
    // The Win32 OpenGL host has no RHI-level self-capture. Honor the shared
    // GEFS_SCREENSHOT_PATH contract in the main loop (45s render gate or an
    // explicit GEFS_SCREENSHOT_WAIT_FOR marker, then retire), so headless CI
    // can grab the presented frame without a visible interactive desktop.
    std::filesystem::path openglScreenshotPath;
    std::filesystem::path openglScreenshotGate;
    const bool openglScreenshotRequested =
        *selection.backend == RhiBackend::OpenGL &&
        std::getenv("GEFS_SCREENSHOT_PATH") != nullptr;
    if (openglScreenshotRequested) {
        openglScreenshotPath = std::getenv("GEFS_SCREENSHOT_PATH");
        if (const auto* gate = std::getenv("GEFS_SCREENSHOT_WAIT_FOR"))
            openglScreenshotGate = gate;
    }
    const auto openglCaptureStart = std::chrono::steady_clock::now();
#endif
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
    // Same-backend reconstruction cycles requested for verification runs
    // (M3.6); executed once after the first presented frame.
    int rhiReconstructCycles = 0;
    for (size_t i = 0; i + 1 < rhiOptions.applicationArguments.size(); ++i) {
        if (std::string(rhiOptions.applicationArguments[i]) ==
            "--rhi-reconstruct") {
            rhiReconstructCycles =
                std::atoi(rhiOptions.applicationArguments[i + 1]);
        }
    }

    RuntimeRoot root;
    auto composed = CreateRhiStartupComposition(
        *selection.backend, config, sceneManager, assetLoader,
        [&root] { return root.GetMemorySystem(); });
    if (!composed.composition) {
        std::cerr << composed.error << '\n';
        return 1;
    }
    BaseApplication& app = *composed.composition->application;
    IGraphicsManager* graphicsManager = composed.composition->graphicsManager.get();
    IPipelineStateManager* pipelineStateManager =
        composed.composition->pipelineStateManager.get();
    auto* platformHost = dynamic_cast<IPlatformHost*>(&app);
    IRenderBackend* renderBackend =
        composed.composition->sessionBackend
            ? composed.composition->sessionBackend.get()
            : dynamic_cast<IRenderBackend*>(&app);
    if (!platformHost || !renderBackend) {
        std::cerr << "Selected backend does not provide a platform host\n";
        return 1;
    }
    root.SetHost(app, *platformHost, *renderBackend);
    auto buildFramePlan = [&] {
        std::vector<IRuntimeModule*> plan = {
            &animationManager, &splashSequence, &assetLoader, graphicsManager,
            &inputManager, &memoryManager, &physicsManager, pipelineStateManager,
            &sceneManager, &gameLogic};
#ifdef DEBUG
        plan.push_back(&debugManager);
#endif
        return plan;
    };
#elif defined(MYGE_USE_VULKAN)
    VulkanPipelineStateManager pipelineStateManager(app.GetRHI(), assetLoader);
    VulkanGraphicsManager graphicsManager(app.GetRHI(), sceneManager, pipelineStateManager);
#elif defined(OS_MACOS)
    TPipelineStateManager pipelineStateManager;
    TGraphicsManager graphicsManager(sceneManager, pipelineStateManager,
                                     [&app] { return (void*)app.GetMetalView(); });
#elif defined(OS_WINDOWS) && !defined(VIEWER_USE_WIN32_OPENGL)
    TPipelineStateManager pipelineStateManager;
    TGraphicsManager graphicsManager(
        app.GetRHI(), sceneManager, pipelineStateManager,
        [&app] { return app.GetMemorySystem(); });
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

#if defined(MYGE_RUNTIME_RHI)
    // Persistent CPU services outlive every render session; the session slot
    // owns the replaceable graphics/pipeline managers. Frame order stays the
    // historical interleaved registration order via buildFramePlan.
    // RegisterManagerModule additionally binds the host's borrowed manager
    // pointers (input routing, resize notification); the application's own
    // module group is never initialized on the root path.
    app.RegisterManagerModule(&animationManager);
    root.RegisterPersistentModule(animationManager);
    app.RegisterManagerModule(&splashSequence);
    root.RegisterPersistentModule(splashSequence);
    app.RegisterManagerModule(&assetLoader);
    root.RegisterPersistentModule(assetLoader);
    app.RegisterManagerModule(&inputManager);
    root.RegisterPersistentModule(inputManager);
    app.RegisterManagerModule(&memoryManager);
    root.RegisterPersistentModule(memoryManager);
    app.RegisterManagerModule(&physicsManager);
    root.RegisterPersistentModule(physicsManager);
    app.RegisterManagerModule(&sceneManager);
    root.RegisterPersistentModule(sceneManager);
    app.RegisterManagerModule(&gameLogic);
    root.RegisterPersistentModule(gameLogic);
#ifdef DEBUG
    app.RegisterManagerModule(&debugManager);
    root.RegisterPersistentModule(debugManager);
#endif
    root.RegisterSessionModule(*graphicsManager);
    root.RegisterSessionModule(*pipelineStateManager);
    app.BindSessionManagers(graphicsManager, pipelineStateManager);
#else
    app.RegisterManagerModule(&animationManager);
    app.RegisterManagerModule(&splashSequence);
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
#if defined(MYGE_RUNTIME_RHI)
    renderBackend->CreateMainWindow();
#else
    app.CreateMainWindow();
#endif

#if defined(MYGE_RUNTIME_RHI)
    ret = root.Initialize();
    if (ret != 0) {
        root.Finalize();
        return ret;
    }
    root.SetFramePlan(buildFramePlan());
#else
    ret = app.Initialize();
    if (ret != 0) {
        app.Finalize();
        return ret;
    }
#endif

    // Drive the modules ahead
#if defined(OS_WEBASSEMBLY)
    auto main_loop = [] { app.Tick(); };

    emscripten_set_main_loop(main_loop, 0, true);
#else
    while (!app.IsQuit()) {
#if defined(MYGE_RUNTIME_RHI)
        root.Tick();
        if (rhiReconstructCycles > 0) {
            const int cycles = rhiReconstructCycles;
            rhiReconstructCycles = 0;
            for (int i = 0; i < cycles; ++i) {
                if (ReconstructRhiSession(
                        root, *composed.composition, sceneManager, assetLoader,
                        [&root] { return root.GetMemorySystem(); }) != 0) {
                    std::cerr << "RHI session reconstruction failed\n";
                    app.RequestQuit();
                    break;
                }
                graphicsManager = composed.composition->graphicsManager.get();
                pipelineStateManager =
                    composed.composition->pipelineStateManager.get();
                root.SetFramePlan(buildFramePlan());
                std::cout << "RHI session reconstructed (cycle " << i + 1
                          << ")\n";
            }
        }
#else
        app.Tick();
#endif
#if defined(OS_WINDOWS) && defined(MYGE_RUNTIME_RHI)
        if (openglScreenshotRequested) {
            const bool ready =
                openglScreenshotGate.empty()
                    ? std::chrono::steady_clock::now() - openglCaptureStart >=
                          std::chrono::seconds(45)
                    : std::filesystem::exists(openglScreenshotGate);
            if (ready) {
                if (CaptureViewerFrame(app, openglScreenshotPath)) {
                    std::cout << "Viewer OpenGL framebuffer capture saved: "
                              << openglScreenshotPath.string() << '\n';
                } else {
                    std::cerr << "Viewer OpenGL framebuffer capture failed\n";
                }
                app.RequestQuit();
            }
        }
#endif
    }
#endif

#if !defined(OS_WEBASSEMBLY)
    // Finalize App
#if defined(MYGE_RUNTIME_RHI)
    root.Finalize();
#else
    app.Finalize();
#endif
#endif

    return ret;
}
