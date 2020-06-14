#include "GuiSubPass.hpp"
#include "imgui/imgui.h"

using namespace My;

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        ImGui::NewFrame();

        ImGui::ShowDemoWindow();

        ImGui::Render();
    }
}