#include "GfxConfiguration.hpp"
#include "config.h"

#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "Bullet/BulletPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "SceneResourceLoader.hpp"
#include "OGEX.hpp"
#include "BilliardGameLogic.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#if defined(OS_WINDOWS) && (defined(MYGE_RUNTIME_RHI) || (!defined(BILLIARD_USE_D3D12) && !defined(MYGE_USE_VULKAN)))
#include "glad/glad.h"
#endif

#if defined(MYGE_RUNTIME_RHI)
#include "RhiComposition.hpp"
#include "Framework/Manager/RuntimeRoot.hpp"
#include "RhiSwitch.hpp"
#if defined(HAS_VULKAN)
#include "RHI/Vulkan/VulkanGraphicsManager.hpp"
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
#if defined(BILLIARD_USE_D3D12)
#include "D3d12Application.hpp"
#else
#include "OpenGLApplication.hpp"
#endif
#else
#include "OpenGLApplication.hpp"
#endif

#if defined(MYGE_USE_VULKAN)
#elif defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#elif defined(OS_WINDOWS)
#if defined(BILLIARD_USE_D3D12)
#include "RHI/D3d/D3d12Config.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif
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

#if defined(OS_WINDOWS) && (defined(MYGE_RUNTIME_RHI) || (!defined(BILLIARD_USE_D3D12) && !defined(MYGE_USE_VULKAN)))
namespace {
bool CaptureBilliardFrame(IApplication& app, const std::filesystem::path& path) {
    uint32_t width = 0, height = 0;
    app.GetFramebufferSize(width, height);
    const uint64_t pitch = (uint64_t(width) * 3 + 3) & ~uint64_t(3);
    const uint64_t size = pitch * height;
    if (!width || !height || size > UINT32_MAX - 54) {
        std::cerr << "Invalid Billiard capture dimensions\n";
        return false;
    }
    std::vector<unsigned char> pixels(size);
    GLint framebuffer, readBuffer, alignment;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_READ_BUFFER, &readBuffer);
    glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
    const auto error = glGetError();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glReadBuffer(readBuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, alignment);
    if (error != GL_NO_ERROR) {
        std::cerr << "Billiard framebuffer capture failed: " << error << '\n';
        return false;
    }
    unsigned char header[54]{};
    auto put32 = [&](int offset, uint32_t value) {
        for (int i = 0; i < 4; ++i) header[offset + i] = value >> (8 * i);
    };
    header[0] = 'B';
    header[1] = 'M';
    put32(2, uint32_t(size) + 54);
    put32(10, 54);
    put32(14, 40);
    put32(18, width);
    put32(22, height);
    header[26] = 1;
    header[28] = 24;
    put32(34, uint32_t(size));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    output.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    output.close();
    if (!output) {
        std::cerr << "Cannot write Billiard capture: " << path << '\n';
        return false;
    }
    return true;
}
}  // namespace
#endif

int main(int argc, char** argv) {
    int ret;
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
    // Billiard's ordinary Windows default is OpenGL; CI selects D3D12 explicitly.
    constexpr RhiBackend defaultBackend =
#if defined(OS_MACOS)
        RhiBackend::Metal;
#else
        RhiBackend::OpenGL;
#endif
    const auto selection = ResolveRhiStartupBackend(
        rhiOptions.requestedBackend, defaultBackend, availability);
    if (!selection.backend) {
        std::cerr << selection.error << '\n';
        return 1;
    }
#endif
    std::filesystem::path regressionDir;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-physicsRegression") == 0) {
            if (++i == argc || !*argv[i]) {
                std::cerr << "-physicsRegression requires an output directory\n";
                return 2;
            }
            regressionDir = argv[i];
        }
    }
    if (!regressionDir.empty()) {
        std::error_code error;
        std::filesystem::create_directories(regressionDir, error);
        if (error) {
            std::cerr << "Cannot create Billiard regression directory: " << error.message() << '\n';
            return 2;
        }
        std::filesystem::remove(regressionDir / "physics-passed", error);
        if (error) {
            std::cerr << "Cannot clear Billiard regression marker: " << error.message() << '\n';
            return 2;
        }
        std::filesystem::remove(regressionDir / "capture-complete", error);
        if (error) {
            std::cerr << "Cannot clear Billiard capture marker: " << error.message() << '\n';
            return 2;
        }
    }

#if defined(MYGE_RUNTIME_RHI)
    const std::string windowTitle = std::string("Billiard - ") +
        std::string(RhiStartupPlatformName()) + "/" +
        std::string(RhiBackendName(*selection.backend)) + " (" +
        std::string(RhiStartupHostName(*selection.backend)) + ")";
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, windowTitle.c_str());
#else
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, "Billiard");
#endif
#if !defined(MYGE_RUNTIME_RHI)
#if defined(MYGE_USE_VULKAN)
    VulkanApplication app(config);
#elif defined(OS_MACOS)
    CocoaMetalApplication app(config);
#elif defined(OS_WINDOWS)
#if defined(BILLIARD_USE_D3D12)
    D3d12Application app(config);
#else
    OpenGLApplication app(config);
#endif
#else
    OpenGLApplication app(config);
#endif
#endif
    MemoryManager memoryManager;
    AssetLoader assetLoader;
    OgexParser sceneParser;
    SceneResourceLoader sceneResources(assetLoader);
    SceneManager sceneManager(assetLoader, sceneParser, sceneResources);
    BulletPhysicsManager physicsManager(sceneManager);
    BilliardGameLogic gameLogic(sceneManager, physicsManager);
    InputManager inputManager(gameLogic);
    AnimationManager animationManager(sceneManager);
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

    // Scheduled cross-backend switches for verification runs (M4):
    // --rhi-switch backend@frame,backend@frame,...
    struct RhiSwitchStep { RhiBackend backend; int frame; };
    std::vector<RhiSwitchStep> rhiSwitchSchedule;
    for (size_t i = 0; i + 1 < rhiOptions.applicationArguments.size(); ++i) {
        if (std::string(rhiOptions.applicationArguments[i]) != "--rhi-switch") {
            continue;
        }
        std::string spec = rhiOptions.applicationArguments[i + 1];
        size_t start = 0;
        while (start < spec.size()) {
            size_t comma = spec.find(',', start);
            if (comma == std::string::npos) comma = spec.size();
            std::string step = spec.substr(start, comma - start);
            size_t at = step.find('@');
            auto backend = at == std::string::npos
                               ? std::nullopt
                               : ParseRhiBackendName(step.substr(0, at));
            if (!backend || at == std::string::npos) {
                std::cerr << "Invalid --rhi-switch step: " << step << '\n';
                return 1;
            }
            rhiSwitchSchedule.push_back(
                {*backend, std::atoi(step.substr(at + 1).c_str())});
            start = comma + 1;
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
            &animationManager, &assetLoader, graphicsManager, &inputManager,
            &memoryManager, &physicsManager, pipelineStateManager, &sceneManager,
            &gameLogic};
#ifdef DEBUG
        plan.push_back(&debugManager);
#endif
        return plan;
    };
    // Simulation pauses during a switch; rendering, input, and asset loading
    // continue (no catch-up interval accumulates).
    auto buildPausedFramePlan = [&] {
        std::vector<IRuntimeModule*> plan = {
            &assetLoader, graphicsManager, &inputManager, &memoryManager,
            pipelineStateManager};
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
#elif defined(OS_WINDOWS) && defined(BILLIARD_USE_D3D12)
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
    // owns the replaceable graphics/pipeline managers. RegisterManagerModule
    // additionally binds the host's borrowed manager pointers (input routing,
    // resize notification); the application's own module group is never
    // initialized on the root path.
    app.RegisterManagerModule(&animationManager);
    root.RegisterPersistentModule(animationManager);
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
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    assetLoader.Initialize();
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

#if defined(MYGE_RUNTIME_RHI)
    renderBackend->CreateMainWindow();
#else
    app.CreateMainWindow();
#endif

#if defined(MYGE_RUNTIME_RHI)
    ret = root.Initialize();
    if (ret != 0) {
        root.Finalize();
        ImGui::DestroyContext();
        return ret;
    }
    root.SetFramePlan(buildFramePlan());
    RhiSwitchController switchController(
        root, *composed.composition, config, sceneManager, assetLoader,
        [&root] { return root.GetMemorySystem(); });
    size_t nextSwitch = 0;
    int rhiFrameCount = 0;
    bool rhiSimulationPaused = false;
#else
    ret = app.Initialize();
    if (ret != 0) {
        app.Finalize();
        ImGui::DestroyContext();
        return ret;
    }
#endif

    if (!regressionDir.empty() &&
        !sceneManager.GetSceneForRendering()->GetFirstCameraNode()) {
        std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: camera missing\n";
#if defined(MYGE_RUNTIME_RHI)
        root.Finalize();
#else
        app.Finalize();
#endif
        ImGui::DestroyContext();
        return 1;
    }

    // Drive the modules ahead
#if defined(OS_WEBASSEMBLY)
    auto main_loop = [] { app.Tick(); };

    emscripten_set_main_loop(main_loop, 0, true);
#else
    int regressionFrame = 0;
    int regressionResult = 1;
    std::map<std::string, Vector3f> initialPositions;
    float cueDisplacement = 0.0f, otherDisplacement = 0.0f;
    while (!app.IsQuit()) {
#if defined(MYGE_RUNTIME_RHI)
        root.Tick();
        ++rhiFrameCount;
        while (nextSwitch < rhiSwitchSchedule.size() &&
               rhiFrameCount >= rhiSwitchSchedule[nextSwitch].frame) {
            const auto target = rhiSwitchSchedule[nextSwitch].backend;
            const auto result = switchController.Request(target);
            std::cout << "RHI switch request " << RhiBackendName(target)
                      << ": " << RhiSwitchRequestResultName(result) << '\n';
            ++nextSwitch;
        }
        switchController.ProcessFrameBoundary();
        if (switchController.ExitRequested()) {
            std::cerr << "RHI switch: " << switchController.LastError() << '\n';
            app.RequestQuit();
            ret = 1;
        }
        if (switchController.ConsumeFramePlanDirty()) {
            // Reconcile held input: keys physically released while the
            // window was gone must not stay pressed in the new session.
            inputManager.Initialize();
            graphicsManager = composed.composition->graphicsManager.get();
            pipelineStateManager =
                composed.composition->pipelineStateManager.get();
            root.SetFramePlan(rhiSimulationPaused ? buildPausedFramePlan()
                                                  : buildFramePlan());
        }
        if (switchController.IsSimulationPaused() != rhiSimulationPaused) {
            rhiSimulationPaused = switchController.IsSimulationPaused();
            root.SetFramePlan(rhiSimulationPaused ? buildPausedFramePlan()
                                                  : buildFramePlan());
        }
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
        if (regressionDir.empty()) continue;
        if (regressionFrame >= 180) {
#if defined(OS_LINUX)
            std::error_code error;
            const bool captured = std::filesystem::exists(regressionDir / "capture-complete", error);
            if (error) {
                std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: " << error.message() << '\n';
                regressionResult = 1;
                app.RequestQuit();
            } else if (captured) {
                app.RequestQuit();
            }
#endif
            continue;
        }
        ++regressionFrame;
        if (regressionFrame < 60) continue;

        auto scene = sceneManager.GetSceneForRendering();
        auto cue = sceneManager.GetSceneGeometryNode("pbb_cue").lock();
        if (!cue) {
            std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: cue ball missing\n";
            app.RequestQuit();
            continue;
        }
        for (const auto& [name, weakNode] : scene->LUT_Name_GeometryNode) {
            if (name.rfind("pbb_", 0) != 0) continue;
            auto node = weakNode.lock();
            if (!node) continue;
            const auto transform = node->GetRenderTransform();
            Vector3f position{transform[3][0], transform[3][1], transform[3][2]};
            if (regressionFrame == 60) {
                initialPositions.emplace(name, position);
            } else {
                const auto delta = position - initialPositions.at(name);
                const float displacement = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
                if (!std::isfinite(displacement) || !std::isfinite(position[2])) {
                    std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: non-finite ball transform\n";
                    app.RequestQuit();
                    break;
                }
                float& maximum = name == "pbb_cue" ? cueDisplacement : otherDisplacement;
                if (displacement > maximum) maximum = displacement;
            }
        }
        if (app.IsQuit()) continue;
        if (regressionFrame == 60) {
#if defined(MYGE_RUNTIME_RHI)
#if defined(OS_WINDOWS)
            if (composed.composition->backend == RhiBackend::OpenGL &&
                !CaptureBilliardFrame(app, regressionDir / "billiard-before.bmp")) {
                app.RequestQuit();
                continue;
            }
#endif
#elif defined(OS_WINDOWS) && !defined(BILLIARD_USE_D3D12) && !defined(MYGE_USE_VULKAN)
            if (!CaptureBilliardFrame(app, regressionDir / "billiard-before.bmp")) {
                app.RequestQuit();
                continue;
            }
#endif
            app.GetGameLogic()->OnLeftKey();
        } else if (regressionFrame == 180) {
            if (cueDisplacement < 0.1f || otherDisplacement < 0.01f) {
                std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: cue=" << cueDisplacement
                          << " other=" << otherDisplacement << '\n';
                app.RequestQuit();
                continue;
            }
#if defined(MYGE_RUNTIME_RHI)
#if defined(OS_WINDOWS)
            if (composed.composition->backend == RhiBackend::OpenGL &&
                !CaptureBilliardFrame(app, regressionDir / "billiard-after.bmp")) {
                app.RequestQuit();
                continue;
            }
#endif
#elif defined(OS_WINDOWS) && !defined(BILLIARD_USE_D3D12) && !defined(MYGE_USE_VULKAN)
            if (!CaptureBilliardFrame(app, regressionDir / "billiard-after.bmp")) {
                app.RequestQuit();
                continue;
            }
#endif
            std::ofstream marker(regressionDir / "physics-passed");
            marker << "cue=" << cueDisplacement << " other=" << otherDisplacement << '\n';
            marker.close();
            if (!marker) {
                std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: cannot write result\n";
                app.RequestQuit();
                continue;
            }
            std::cout << "BILLIARD_PHYSICS_REGRESSION PASSED: cue=" << cueDisplacement
                      << " other=" << otherDisplacement << std::endl;
            regressionResult = 0;
#if defined(MYGE_RUNTIME_RHI)
            // Capture/quit dispatch follows the active backend, not the build.
            // Linux completion is driven by the external capture script's
            // capture-complete marker regardless of backend. D3D12/Metal capture
            // paths retire the application themselves once the frame is saved.
#if !defined(OS_LINUX)
            if (composed.composition->backend == RhiBackend::Vulkan) {
#if defined(HAS_VULKAN)
                if (!static_cast<VulkanGraphicsManager*>(graphicsManager)->HasPendingScreenshot())
                    app.RequestQuit();
#endif
            }
#if defined(OS_WINDOWS)
            else if (composed.composition->backend == RhiBackend::OpenGL) {
                app.RequestQuit();
            }
#endif
#endif
#elif defined(MYGE_USE_VULKAN) && !defined(OS_LINUX)
            // Linux completion is driven by the external capture script's
            // capture-complete marker. Elsewhere quit automatically unless a
            // screenshot request is still pending; the capture path retires the
            // application itself once the frame is saved.
            if (!graphicsManager.HasPendingScreenshot()) app.RequestQuit();
#elif defined(OS_WINDOWS) && !defined(BILLIARD_USE_D3D12) && !defined(MYGE_USE_VULKAN)
            app.RequestQuit();
#endif
        }
    }
    if (!regressionDir.empty()) ret = regressionResult;
#endif

#if !defined(OS_WEBASSEMBLY)
    // Finalize App
#if defined(MYGE_RUNTIME_RHI)
    root.Finalize();
#else
    app.Finalize();
#endif
#endif

    ImGui::DestroyContext();

    return ret;
}
