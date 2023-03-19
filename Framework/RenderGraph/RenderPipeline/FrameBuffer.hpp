#include "RenderTarget.hpp"


namespace My::RenderGraph {
    struct FrameBuffer {
        RenderTarget	color_attachment;

        RenderTarget	depth_attachment;


        void reflectMembers() {
            if (ImGui::TreeNode(&color_attachment, "color_attachment")) {
                color_attachment.reflectMembers();
                ImGui::TreePop();
            }

            if (ImGui::TreeNode(&depth_attachment, "depth_attachment")) {
                depth_attachment.reflectMembers();
                ImGui::TreePop();
            }

        }

        void reflectUI() {
            ImGui::Begin("FrameBuffer");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
