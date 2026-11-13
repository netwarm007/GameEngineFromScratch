#pragma once
#include "geommath.hpp"
#include "imgui/imgui.h"
#include "RenderTarget.hpp"




namespace My::RenderGraph {
    struct FrameBuffer {
        std::vector<RenderTarget>	color_attachments;

        Vector3f	color_clear_value;

        RenderTarget	depth_attachment;

        float	depth_clear_value;


        void reflectMembers() {
            for (int i = 0; i < color_attachments.size(); i++) {
                if (ImGui::TreeNode(&color_attachments[i], "color_attachments[%d]", i)) {
                    color_attachments[i].reflectMembers();
                    ImGui::TreePop();
                }
            }

            ImGui::ColorEdit3( "color_clear_value", color_clear_value.data );

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
