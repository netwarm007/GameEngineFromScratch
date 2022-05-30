#include "GuiSubPass.hpp"
#include "imgui/imgui.h"
#include "RenderPipeline.hpp"

#include <climits>
#include <cstdlib>
#include <cstdio>

using namespace My;

static RenderPipeline pipeline;

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        ImGui::NewFrame();

        ImGui::ShowAboutWindow();

        pipeline.reflectUI();

        ImGui::Render();

        ImGui::EndFrame();
    }
}