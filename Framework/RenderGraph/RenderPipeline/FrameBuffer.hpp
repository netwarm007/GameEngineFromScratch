#pragma once
#include "geommath.hpp"
using color = My::Vector4f;
#include "RenderTarget.hpp"




namespace My::RenderGraph {
    struct FrameBuffer {
        RenderTarget	color_attachment;

        color	color_clear_value;

        RenderTarget	depth_attachment;

        float	depth_clear_value;


        void reflectMembers() {
            if (ImGui::TreeNode(&color_attachment, "color_attachment")) {
                color_attachment.reflectMembers();
                ImGui::TreePop();
            }

            ImGui::ColorEdit4( "color_clear_value", color_clear_value.data );

            if (ImGui::TreeNode(&depth_attachment, "depth_attachment")) {
                depth_attachment.reflectMembers();
                ImGui::TreePop();
            }

            ImGui::SliderFloat( "depth_clear_value", &depth_clear_value, 0.0f, 1.0f );

        }

        void reflectUI() {
            ImGui::Begin("FrameBuffer");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
