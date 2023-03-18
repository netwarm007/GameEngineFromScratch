#include "FrameBuffer.hpp"
#include "PipelineState.hpp"

namespace My {
    struct RenderPass {
        FrameBuffer	frame_buffer;

        PipelineState	pipeline_state;


        void reflectMembers() {
            ImGui::Text("frame_buffer");
            frame_buffer.reflectMembers();

            ImGui::Text("pipeline_state");
            pipeline_state.reflectMembers();

        }

        void reflectUI() {
            ImGui::Begin("RenderPass");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My
