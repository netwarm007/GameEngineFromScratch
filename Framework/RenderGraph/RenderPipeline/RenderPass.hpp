#include "FrameBuffer.hpp"
#include "PipelineState.hpp"

namespace My::RenderGraph {
    struct RenderPass {
        FrameBuffer	frame_buffer;

        PipelineState	pipeline_state;


        void reflectMembers() {
            if (ImGui::TreeNode(&frame_buffer, "frame_buffer")) {
                frame_buffer.reflectMembers();
                ImGui::TreePop();
            }

            if (ImGui::TreeNode(&pipeline_state, "pipeline_state")) {
                pipeline_state.reflectMembers();
                ImGui::TreePop();
            }

        }

        void reflectUI() {
            ImGui::Begin("RenderPass");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
