#include "AssetLoader.hpp"
#include "Platform/CrossPlatformGfxApp.hpp"
#include "Platform/TGraphicsManager.hpp"

#include "imgui/imgui.h"

#include "RenderGraph/RenderPipeline/RenderPipeline.hpp"

using namespace My;

RenderGraph::RenderPipeline renderPipeline;

int test(BaseApplication& app) {
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

    app.CreateMainWindow();

    auto result = app.Initialize();

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
        static bool show_demo_window = false;
        auto& frame = m_Frames[m_nFrameIndex];
        frame.clearColor = renderPipeline.render_passes[0].frame_buffer.color_clear_value;

        BeginPass(frame);

        // GUI
        if (show_demo_window) {
            ImGui::ShowDemoWindow(&show_demo_window);
        }

        {
            ImGui::Begin((const char*)u8"图形管道测试");
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
                            "Basic Graphics Window Test");

    RenderGraph::RenderPass render_pass;
    render_pass.frame_buffer.color_clear_value = { 0.2f, 0.3f, 0.4f, 1.0f };
    render_pass.frame_buffer.color_attachment.format = RenderGraph::TextureFormat::R8G8B8A8_UNORM;
    render_pass.frame_buffer.color_attachment.width = config.screenWidth;
    render_pass.frame_buffer.color_attachment.height = config.screenHeight;
    render_pass.frame_buffer.color_attachment.scale_x = 1.0f;
    render_pass.frame_buffer.color_attachment.scale_y = 1.0f;
    render_pass.frame_buffer.color_attachment.load_action = RenderGraph::RenderTargetLoadStoreAction::Clear;
    render_pass.frame_buffer.color_attachment.store_action = RenderGraph::RenderTargetLoadStoreAction::Keep;

    render_pass.frame_buffer.depth_clear_value = 0.0f;
    render_pass.frame_buffer.depth_attachment.format = RenderGraph::TextureFormat::D32_FLOAT;
    render_pass.frame_buffer.depth_attachment.width = config.screenWidth;
    render_pass.frame_buffer.depth_attachment.height = config.screenHeight;
    render_pass.frame_buffer.depth_attachment.scale_x = 1.0f;
    render_pass.frame_buffer.depth_attachment.scale_y = 1.0f;
    render_pass.frame_buffer.depth_attachment.load_action = RenderGraph::RenderTargetLoadStoreAction::Clear;
    render_pass.frame_buffer.depth_attachment.store_action = RenderGraph::RenderTargetLoadStoreAction::DontCare;

    render_pass.pipeline_state.topology_type = RenderGraph::TopologyType::Triangle;
    render_pass.pipeline_state.blend_state.enable = false;
    render_pass.pipeline_state.depth_stencil_state.enable = false;
    render_pass.pipeline_state.rasterizer_state.conservative = false;
    render_pass.pipeline_state.rasterizer_state.cull_mode = RenderGraph::CullMode::Back;
    render_pass.pipeline_state.rasterizer_state.depth_bias = 0;
    render_pass.pipeline_state.rasterizer_state.depth_bias_clamp = 0;
    render_pass.pipeline_state.rasterizer_state.depth_clip_enabled = false;
    render_pass.pipeline_state.rasterizer_state.fill_mode = RenderGraph::FillMode::Solid;
    render_pass.pipeline_state.rasterizer_state.front_counter_clockwise = true;
    render_pass.pipeline_state.rasterizer_state.multisample_enabled = true;
    render_pass.pipeline_state.rasterizer_state.slope_scaled_depth_bias = 0;

    renderPipeline.render_passes.emplace_back(render_pass);

    auto app = My::CreateApplication(config);
    TestGraphicsManager graphicsManager;

    app->SetCommandLineParameters(argc, argv);
    app->RegisterManagerModule(&graphicsManager);

    result |= test(*app);

    delete app;

    return result;
}