#pragma once
#include "FillMode.hpp"
#include "CullMode.hpp"








namespace My::RenderGraph {
    struct RasterizerState {
        FillMode::Enum	fill_mode;

        CullMode::Enum	cull_mode;

        bool	front_counter_clockwise;

        int32_t	depth_bias;

        float	depth_bias_clamp;

        float	slope_scaled_depth_bias;

        bool	depth_clip_enabled;

        bool	multisample_enabled;

        bool	conservative;


        void reflectMembers() {
            ImGui::PushID(&fill_mode);
            ImGui::Combo( "fill_mode", (int32_t*)&fill_mode, FillMode::s_value_names, FillMode::Count );
            ImGui::PopID();

            ImGui::PushID(&cull_mode);
            ImGui::Combo( "cull_mode", (int32_t*)&cull_mode, CullMode::s_value_names, CullMode::Count );
            ImGui::PopID();

            ImGui::PushID(&front_counter_clockwise);
            ImGui::Checkbox( "front_counter_clockwise", &front_counter_clockwise );
            ImGui::PopID();

            ImGui::PushID(&depth_bias);
            ImGui::InputScalar( "depth_bias", ImGuiDataType_S32, &depth_bias );
            ImGui::PopID();

            ImGui::PushID(&depth_bias_clamp);
            ImGui::SliderFloat( "depth_bias_clamp", &depth_bias_clamp, 0.0f, 1.0f );
            ImGui::PopID();

            ImGui::PushID(&slope_scaled_depth_bias);
            ImGui::SliderFloat( "slope_scaled_depth_bias", &slope_scaled_depth_bias, 0.0f, 1.0f );
            ImGui::PopID();

            ImGui::PushID(&depth_clip_enabled);
            ImGui::Checkbox( "depth_clip_enabled", &depth_clip_enabled );
            ImGui::PopID();

            ImGui::PushID(&multisample_enabled);
            ImGui::Checkbox( "multisample_enabled", &multisample_enabled );
            ImGui::PopID();

            ImGui::PushID(&conservative);
            ImGui::Checkbox( "conservative", &conservative );
            ImGui::PopID();

        }

        void reflectUI() {
            ImGui::Begin("RasterizerState");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
