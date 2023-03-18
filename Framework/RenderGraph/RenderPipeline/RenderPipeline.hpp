#include "RenderPass.hpp"
#include <vector>

namespace My {
    struct RenderPipeline {
        std::vector<RenderPass>	render_passes;


        void reflectMembers() {
            for (int i = 0; i < render_passes.size(); i++) {
                ImGui::Text("render_passes[%s]", i);
                render_passes[i].reflectMembers();
            }

        }

        void reflectUI() {
            ImGui::Begin("RenderPipeline");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My
