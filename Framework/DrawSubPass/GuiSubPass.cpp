#include "GuiSubPass.hpp"
#include "imgui/imgui.h"

#include <deque>

using namespace My;

void GuiSubPass::Draw(Frame& frame) {
    static std::deque<float> fps_data;
    static float max_fps = 0.0f;

    if (ImGui::GetCurrentContext()) {
        ImGui::Begin((const char*)u8"你好，引擎！");  // Create a window called "Hello, Engine!" and append into it.

        ImGui::ColorEdit3(
            (const char*)u8"清屏色",
            (float*)frame.clearColor);  // Edit 3 floats representing a color

        float fps = ImGui::GetIO().Framerate;
        ImGui::Text((const char*)u8"平均渲染时间 %.3f 毫秒/帧 (%.1f FPS)",
                    1000.0f / fps,
                    fps);
        
        fps_data.push_back(fps);
        if (fps_data.size() > 100) fps_data.pop_front();
        if (fps > max_fps) max_fps = fps;

        auto getData = [](void* data, int index)->float {
            return ((decltype(fps_data)*)data)->at(index);
        };

        ImGui::PlotLines((const char*)u8"帧率", getData, (void *)&fps_data, fps_data.size(), 0, "FPS", 0, max_fps);

        ImGui::End();
    }
}