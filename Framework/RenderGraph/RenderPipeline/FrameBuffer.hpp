#include "RenderTarget.hpp"


namespace My::RenderGraph {
    struct FrameBuffer {
        RenderTarget	color_attachment;

        RenderTarget	depth_attachment;


        void reflectMembers() {
            ImGui::PushID(&color_attachment);
            ImGui::Text("color_attachment");
            color_attachment.reflectMembers();
            ImGui::PopID();

            ImGui::PushID(&depth_attachment);
            ImGui::Text("depth_attachment");
            depth_attachment.reflectMembers();
            ImGui::PopID();

        }

        void reflectUI() {
            ImGui::Begin("FrameBuffer");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
