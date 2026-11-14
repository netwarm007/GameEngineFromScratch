#include "GfxConfiguration.hpp"
#include "config.h"

#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "Bullet/BulletPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "OGEX.hpp"
#include "BilliardGameLogic.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#if defined(OS_WINDOWS) || defined(OS_LINUX)
#include "glad/glad.h"
#endif

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#elif defined(OS_WINDOWS)
//#include "D3d12Application.hpp"
#include "OpenGLApplication.hpp"
#else
#include "OpenGLApplication.hpp"
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

#if defined(OS_WINDOWS) || defined(OS_LINUX)
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
    }

    GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 1024, 768, "Billiard");
    BilliardGameLogic gameLogic;
    BulletPhysicsManager physicsManager;
#if defined(OS_MACOS)
    CocoaMetalApplication app(config);
#elif defined(OS_WINDOWS)
    // D3d12Application app(config);
    OpenGLApplication app(config);
#else
    OpenGLApplication app(config);
#endif
    MemoryManager memoryManager;
    AssetLoader assetLoader;
    OgexParser sceneParser;
    SceneManager sceneManager(assetLoader, sceneParser);
    InputManager inputManager;
    AnimationManager animationManager(sceneManager);
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

    app.CreateMainWindow();

    ret = app.Initialize();
    if (ret != 0) {
        app.Finalize();
        ImGui::DestroyContext();
        return ret;
    }

    if (!regressionDir.empty() &&
        !sceneManager.GetSceneForRendering()->GetFirstCameraNode()) {
        std::cerr << "BILLIARD_PHYSICS_REGRESSION FAILED: camera missing\n";
        app.Finalize();
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
        app.Tick();
        if (regressionDir.empty() || regressionFrame >= 180) continue;
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
#if defined(OS_WINDOWS) || defined(OS_LINUX)
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
#if defined(OS_WINDOWS) || defined(OS_LINUX)
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
#if defined(OS_WINDOWS) || defined(OS_LINUX)
            app.RequestQuit();
#endif
        }
    }
    if (!regressionDir.empty()) ret = regressionResult;
#endif

#if !defined(OS_WEBASSEMBLY)
    // Finalize App
    app.Finalize();
#endif

    ImGui::DestroyContext();

    return ret;
}
