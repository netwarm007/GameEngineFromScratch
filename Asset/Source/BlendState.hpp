#pragma once


#include "RenderTargetBlend.hpp"








namespace My {
    struct BlendState {
        bool	enable;

        bool	separate_blend;

        RenderTargetBlend	render_target_blend0;

        RenderTargetBlend	render_target_blend1;

        RenderTargetBlend	render_target_blend2;

        RenderTargetBlend	render_target_blend3;

        RenderTargetBlend	render_target_blend4;

        RenderTargetBlend	render_target_blend5;

        RenderTargetBlend	render_target_blend6;

        RenderTargetBlend	render_target_blend7;


        void reflectMembers() {
            ImGui::Checkbox( "enable", &enable );
            ImGui::Checkbox( "separate_blend", &separate_blend );
            ImGui::Text("render_target_blend0");
            render_target_blend0.reflectMembers();

            ImGui::Text("render_target_blend1");
            render_target_blend1.reflectMembers();

            ImGui::Text("render_target_blend2");
            render_target_blend2.reflectMembers();

            ImGui::Text("render_target_blend3");
            render_target_blend3.reflectMembers();

            ImGui::Text("render_target_blend4");
            render_target_blend4.reflectMembers();

            ImGui::Text("render_target_blend5");
            render_target_blend5.reflectMembers();

            ImGui::Text("render_target_blend6");
            render_target_blend6.reflectMembers();

            ImGui::Text("render_target_blend7");
            render_target_blend7.reflectMembers();

        }

        void reflectUI() {
            ImGui::Begin("BlendState");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My
