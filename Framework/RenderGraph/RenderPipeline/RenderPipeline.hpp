#include "RenderPass.hpp"
#include <vector>

namespace My::RenderGraph {
    struct RenderPipeline {
        std::vector<RenderPass>	render_passes;


        void reflectMembers() {
            ImGui::PushID(&render_passes);
            for (int i = 0; i < render_passes.size(); i++) {
                ImGui::PushID(i);
                ImGui::Text("render_passes[%d]", i);
                render_passes[i].reflectMembers();
                ImGui::PopID();
            }
            ImGui::PopID();

        }

        void reflectUI() {
            ImGui::Begin("RenderPipeline");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
