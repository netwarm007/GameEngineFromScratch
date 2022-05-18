#include "GuiSubPass.hpp"
#include "imgui/imgui.h"

using namespace My;

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        ImGui::NewFrame();

        ImGui::Begin("Debug Window", nullptr, ImGuiWindowFlags_AlwaysAutoResize
                                                        | ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::Text("Hello! I'm a Game Engine From Scratch!");
        ImGui::End();

        ImGui::Render();

        ImGui::EndFrame();
    }
}