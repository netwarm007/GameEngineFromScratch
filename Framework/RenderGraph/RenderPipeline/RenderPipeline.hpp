#pragma once
#include "geommath.hpp"
#include "imgui/imgui.h"
#include "RenderPass.hpp"
#include <vector>

namespace My::RenderGraph {
    struct RenderPipeline {
        std::vector<RenderPass>	render_passes;


        void reflectMembers() {
            for (int i = 0; i < render_passes.size(); i++) {
                if (ImGui::TreeNode(&render_passes[i], "render_passes[%d]", i)) {
                    render_passes[i].reflectMembers();
                    ImGui::TreePop();
                }
            }

        }

        void reflectUI() {
            ImGui::Begin("RenderPipeline");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
