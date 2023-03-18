#include "FrameBuffer.hpp"
#include "PipelineState.hpp"

namespace My::RenderGraph {
    struct RenderPass {
        FrameBuffer	frame_buffer;

        PipelineState	pipeline_state;


        void reflectMembers() {
            ImGui::PushID(&frame_buffer);
            ImGui::Text("frame_buffer");
            frame_buffer.reflectMembers();
            ImGui::PopID();

            ImGui::PushID(&pipeline_state);
            ImGui::Text("pipeline_state");
            pipeline_state.reflectMembers();
            ImGui::PopID();

        }

        void reflectUI() {
            ImGui::Begin("RenderPass");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
