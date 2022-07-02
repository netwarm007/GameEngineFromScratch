#include "GuiSubPass.hpp"
#include "imgui/imgui.h"

#include <climits>
#include <cstdlib>
#include <cstdio>

using namespace My;

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin((const char*)u8"你好，引擎！");  // Create a window called "Hello,
                                            // world!" and append into it.

            ImGui::Text(
                (const char*)u8"这里有一些重要信息。");  // Display some text (you can use
                                               // a format strings too)
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
    }
}