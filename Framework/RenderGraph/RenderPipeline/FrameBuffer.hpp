#include "RenderTarget.hpp"


namespace My {
    struct FrameBuffer {
        RenderTarget	color_attachment;

        RenderTarget	depth_attachment;


        void reflectMembers() {
            ImGui::Text("color_attachment");
            color_attachment.reflectMembers();

            ImGui::Text("depth_attachment");
            depth_attachment.reflectMembers();

        }

        void reflectUI() {
            ImGui::Begin("FrameBuffer");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My
