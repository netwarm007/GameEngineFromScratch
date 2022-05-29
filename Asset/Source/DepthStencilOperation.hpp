#pragma once
#include "StencilOperation.hpp"




namespace My {
    struct DepthStencilOperation {
        StencilOperation::Enum	fail;

        StencilOperation::Enum	depth_fail;

        StencilOperation::Enum	pass;

        ComparisonFunction::Enum	func;


        void reflectMembers() {
            StencilOperation::Enum fail;
            ImGui::Combo( "fail", (int32_t*)&fail, StencilOperation::s_value_names, StencilOperation::Count );

            StencilOperation::Enum depth_fail;
            ImGui::Combo( "depth_fail", (int32_t*)&depth_fail, StencilOperation::s_value_names, StencilOperation::Count );

            StencilOperation::Enum pass;
            ImGui::Combo( "pass", (int32_t*)&pass, StencilOperation::s_value_names, StencilOperation::Count );

            ComparisonFunction::Enum func;
            ImGui::Combo( "func", (int32_t*)&func, ComparisonFunction::s_value_names, ComparisonFunction::Count );

        }

        void reflectUI() {
            ImGui::Begin("DepthStencilOperation");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My
