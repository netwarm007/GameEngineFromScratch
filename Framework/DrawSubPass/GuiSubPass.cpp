#include "GuiSubPass.hpp"
#include "imgui/imgui.h"

#include <climits>
#include <cstdlib>
#include <cstdio>

using namespace My;

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        ImGui::NewFrame();

        ImGui::ShowAboutWindow();

        ImGui::Render();

        ImGui::EndFrame();
    }
}