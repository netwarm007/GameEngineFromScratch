#pragma once

#include "Blend.hpp"

#include "BlendOperation.hpp"





namespace My::RenderGraph {
    struct RenderTargetBlend {
        bool	blend_enable;

        Blend::Enum	src_blend;

        Blend::Enum	dst_blend;

        BlendOperation::Enum	blend_operation;

        Blend::Enum	src_blend_alpha;

        Blend::Enum	dst_blend_alpha;

        BlendOperation::Enum	blend_operation_alpha;

        char	color_write_mask;


        void reflectMembers() {
            ImGui::PushID(&blend_enable);
            ImGui::Checkbox( "blend_enable", &blend_enable );
            ImGui::PopID();

            ImGui::PushID(&src_blend);
            ImGui::Combo( "src_blend", (int32_t*)&src_blend, Blend::s_value_names, Blend::Count );
            ImGui::PopID();

            ImGui::PushID(&dst_blend);
            ImGui::Combo( "dst_blend", (int32_t*)&dst_blend, Blend::s_value_names, Blend::Count );
            ImGui::PopID();

            ImGui::PushID(&blend_operation);
            ImGui::Combo( "blend_operation", (int32_t*)&blend_operation, BlendOperation::s_value_names, BlendOperation::Count );
            ImGui::PopID();

            ImGui::PushID(&src_blend_alpha);
            ImGui::Combo( "src_blend_alpha", (int32_t*)&src_blend_alpha, Blend::s_value_names, Blend::Count );
            ImGui::PopID();

            ImGui::PushID(&dst_blend_alpha);
            ImGui::Combo( "dst_blend_alpha", (int32_t*)&dst_blend_alpha, Blend::s_value_names, Blend::Count );
            ImGui::PopID();

            ImGui::PushID(&blend_operation_alpha);
            ImGui::Combo( "blend_operation_alpha", (int32_t*)&blend_operation_alpha, BlendOperation::s_value_names, BlendOperation::Count );
            ImGui::PopID();

            ImGui::PushID(&color_write_mask);
            ImGui::InputScalar( "color_write_mask", ImGuiDataType_S8, &color_write_mask );
            ImGui::PopID();

        }

        void reflectUI() {
            ImGui::Begin("RenderTargetBlend");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
