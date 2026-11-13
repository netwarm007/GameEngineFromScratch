#pragma once
#include "geommath.hpp"
#include "imgui/imgui.h"




#include "RenderTargetLoadStoreAction.hpp"

#include "TextureFormat.hpp"

namespace My::RenderGraph {
    struct RenderTarget {
        uint16_t	width;

        uint16_t	height;

        float	scale_x;

        float	scale_y;

        RenderTargetLoadStoreAction::Enum	load_action;

        RenderTargetLoadStoreAction::Enum	store_action;

        TextureFormat::Enum	format;


        void reflectMembers() {
            ImGui::InputScalar( "width", ImGuiDataType_U16, &width );

            ImGui::InputScalar( "height", ImGuiDataType_U16, &height );

            ImGui::SliderFloat( "scale_x", &scale_x, 0.0f, 1.0f );

            ImGui::SliderFloat( "scale_y", &scale_y, 0.0f, 1.0f );

            ImGui::Combo( "load_action", (int32_t*)&load_action, RenderTargetLoadStoreAction::s_value_names, RenderTargetLoadStoreAction::Count );

            ImGui::Combo( "store_action", (int32_t*)&store_action, RenderTargetLoadStoreAction::s_value_names, RenderTargetLoadStoreAction::Count );

            ImGui::Combo( "format", (int32_t*)&format, TextureFormat::s_value_names, TextureFormat::Count );

        }

        void reflectUI() {
            ImGui::Begin("RenderTarget");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
