#pragma once




#include "TextureFormat.hpp"

namespace My::RenderGraph {
    struct RenderTarget {
        uint16_t	width;

        uint16_t	height;

        float	scale_x;

        float	scale_y;

        TextureFormat::Enum	format;


        void reflectMembers() {
            ImGui::InputScalar( "width", ImGuiDataType_U16, &width );

            ImGui::InputScalar( "height", ImGuiDataType_U16, &height );

            ImGui::SliderFloat( "scale_x", &scale_x, 0.0f, 1.0f );

            ImGui::SliderFloat( "scale_y", &scale_y, 0.0f, 1.0f );

            ImGui::Combo( "format", (int32_t*)&format, TextureFormat::s_value_names, TextureFormat::Count );

        }

        void reflectUI() {
            ImGui::Begin("RenderTarget");
            reflectMembers();
            ImGui::End();
        }
    };
} // namespace My::RenderGraph
