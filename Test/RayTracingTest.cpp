#include "Platform/CrossPlatformGfxApp.hpp"
#include "Platform/TGraphicsManager.hpp"

#include "AssetLoader.hpp"

#include "imgui/imgui.h"

#include "BVH.hpp"
#include "Encoder/PPM.hpp"
#include "Image.hpp"
#include "RayTracingCamera.hpp"
#include "geommath.hpp"
#include "portable.hpp"

#define float_precision float
#include "TestMaterial.hpp"
#include "TestScene.hpp"

#include "PathTracing.hpp"

#include <chrono>
#include <sstream>
#include <string_view>

std::ostringstream oss;
bool is_closed = false;

using image = My::Image;
using bvh = My::BVHNode<float_precision>;
using camera = My::RayTracingCamera<float_precision>;

using namespace My;

void gui_loop(BaseApplication& app) {
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
    builder.AddText((const char*)u8"屏擎渲帧钮");

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

        is_closed = true;

        app.Finalize();
    }

    ImGui::DestroyContext();
}

class TestGraphicsManager : public TGraphicsManager {
    void Draw() override {
        static bool show_demo_window = true;
        auto& frame = m_Frames[m_nFrameIndex];
        BeginPass(frame);

        {
            ImGui::Begin(
                (const char*)u8"光线追踪（路径追踪）");  // Create a window
                                                         // called "Hello,
                                                         // world!" and append
                                                         // into it.
            size_t start_pos = 0;
            size_t end_pos = 0;
            std::string str(oss.str());
            do {
                start_pos = end_pos + 1;
                end_pos = str.find('\n', start_pos);
                start_pos = str.rfind('\r', end_pos);
                if (start_pos == std::string::npos) start_pos = 0;
                ImGui::Text(str.substr(start_pos, end_pos).c_str());
            } while (end_pos != std::string::npos);

            ImGui::End();
        }
        EndPass(frame);
    }
};

// Main
int main(int argc, char** argv) {
    GfxConfiguration gui_config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "Ray Tracing Test");

    auto pApp = CreateApplication(gui_config);

    TestGraphicsManager graphicsManager;

    pApp->SetCommandLineParameters(argc, argv);
    pApp->RegisterManagerModule(&graphicsManager);

    auto gui_task = [pApp]() { gui_loop(*pApp); };
    auto gui_future = std::async(std::launch::async, gui_task);

    // Render Settings
    const int image_width = gui_config.screenWidth;
    const int image_height = gui_config.screenHeight;
    const float_precision aspect_ratio = (float_precision)image_width / (float_precision)image_height;
    const My::raytrace_config config = {.samples_per_pixel = 512,
                                        .max_depth = 50};

    // World
    auto world = random_scene();
    bvh world_bvh(world);

    // Camera
    point3 lookfrom({13, 2, 3});
    point3 lookat({0, 0, 0});
    vec3 vup({0, 1, 0});
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, (float_precision)20.0, aspect_ratio,
               aperture, dist_to_focus);

    // Canvas
    image img;
    img.Width = image_width;
    img.Height = image_height;
    img.bitcount = 24;
    img.bitdepth = 8;
    img.pixel_format = My::PIXEL_FORMAT::RGB8;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = My::COMPRESSED_FORMAT::NONE;
    img.data_size = img.Width * img.Height * (img.bitcount >> 3);
    img.data = new uint8_t[img.data_size];

    // Render
    auto start = std::chrono::steady_clock::now();
    My::PathTracing<float_precision>::raytrace(config, world_bvh, cam, img,
                                               oss, is_closed);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << "Rendering time: " << diff.count() << " s";

    gui_future.wait();

    delete pApp;

#if 0
    My::PpmEncoder encoder;
    encoder.Encode(img);
#endif
    img.SaveTGA("raytraced.tga");

    return 0;
}
