#pragma once
#include "StencilOperation.hpp"




namespace My::RenderGraph {
    struct DepthStencilOperation {
        StencilOperation::Enum	fail;

        StencilOperation::Enum	depth_fail;

        StencilOperation::Enum	pass;

        ComparisonFunction::Enum	func;


        void reflectMembers() {
            ImGui::Combo( "fail", (int32_t*)&fail, StencilOperation::s_value_names, StencilOperation::Count );

            ImGui::Combo( "depth_fail", (int32_t*)&depth_fail, StencilOperation::s_value_names, StencilOperation::Count );

            ImGui::Combo( "pass", (int32_t*)&pass, StencilOperation::s_value_names, StencilOperation::Count );

            ImGui::Combo( "func", (int32_t*)&func, ComparisonFunction::s_value_names, ComparisonFunction::Count );

        }

        void reflectUI() {
            ImGui::Begin("DepthStencilOperation");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
