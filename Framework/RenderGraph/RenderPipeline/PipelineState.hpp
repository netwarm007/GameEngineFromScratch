#pragma once
#include "BlendState.hpp"
#include "RasterizerState.hpp"
#include "DepthStencilState.hpp"
#include "TopologyType.hpp"

namespace My::RenderGraph {
    struct PipelineState {
        BlendState	blend_state;

        RasterizerState	rasterizer_state;

        DepthStencilState	depth_stencil_state;

        TopologyType::Enum	topology_type;


        void reflectMembers() {
            ImGui::PushID(&blend_state);
            ImGui::Text("blend_state");
            blend_state.reflectMembers();
            ImGui::PopID();

            ImGui::PushID(&rasterizer_state);
            ImGui::Text("rasterizer_state");
            rasterizer_state.reflectMembers();
            ImGui::PopID();

            ImGui::PushID(&depth_stencil_state);
            ImGui::Text("depth_stencil_state");
            depth_stencil_state.reflectMembers();
            ImGui::PopID();

            ImGui::PushID(&topology_type);
            ImGui::Combo( "topology_type", (int32_t*)&topology_type, TopologyType::s_value_names, TopologyType::Count );
            ImGui::PopID();

        }

        void reflectUI() {
            ImGui::Begin("PipelineState");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
