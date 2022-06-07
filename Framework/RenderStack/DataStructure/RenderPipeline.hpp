#include "PipelineState.hpp"

namespace My {
    struct RenderPipeline {
        PipelineState	state;


        void reflectMembers() {
            ImGui::Text("state");
            state.reflectMembers();

        }

        void reflectUI() {
            ImGui::Begin("RenderPipeline");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My
